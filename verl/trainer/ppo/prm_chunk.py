from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto

logger = logging.getLogger(__name__)

THINKPRM_VARIANT = "thinkprm"
SUCCESS_PRM_VARIANT = "success_prm"
CUSTOM_PRM_VARIANT = "custom"
SUPPORTED_PRM_VARIANTS = {THINKPRM_VARIANT, SUCCESS_PRM_VARIANT, CUSTOM_PRM_VARIANT}

THINKPRM_SCORING_PROMPT_TEMPLATE = """You are given a math problem and a proposed multiple-step solution (with a step on each line):
[Math Problem]
{problem}
[Solution]
{solution}
Review and critique the proposed solution steps and determine whether each step is correct. If the solution is incomplete, only critique the steps that are provided. Your output must be in the following format:
Let's verify step by step:
Step 1: <critique>...The step is \boxed{{correct/incorrect}}
Step 2: <critique>...The step is \boxed{{correct/incorrect}}
. . .
Step n: <critique>...The step is \boxed{{correct/incorrect}}
Once you find an incorrect step, you should stop since you don't need to analyze the remaining steps."""

SUCCESS_PRM_SCORING_PROMPT_TEMPLATE = """You are given a math problem and a proposed multiple-step solution, with one logical step on each numbered line.

[Math Problem]
{problem}

[Solution]
{solution}


Review and critique the proposed solution steps and determine whether each step should be labeled success or fail.

A success step is one that is logically correct and brings the solution closer to solving the problem or reaching the correct final answer.

A fail step is one that is logically incorrect, irrelevant, misleading, or does not bring the solution closer to the correct final answer.

Also rate each step for its likelihood of success on a scale from 1 to 10:
10 = this step definitely leads toward success.
5 = this step might lead toward success.
1 = this step does not lead toward success.

If the solution is incomplete, only critique the steps that are provided.

Your output must be in the following format:

Let's verify step by step:

Step 1: <critique> This step is \boxed{{success}} and scores \boxed{{rating}}
Step 2: <critique> This step is \boxed{{fail}} and scores \boxed{{rating}}
...
Step n: <critique> This step is \boxed{{success/fail}} and scores \boxed{{rating}}

If you find a fail step, you should still try to evaluate the next steps for success whenever possible, because later steps may recover or still move closer to the solution. However, if an earlier logical error makes the following steps impossible to evaluate meaningfully, state that clearly."""

DEFAULT_SCORING_PROMPT_TEMPLATE = THINKPRM_SCORING_PROMPT_TEMPLATE
DEFAULT_DECISION_PREFIX = "This step is \\boxed{"

PRM_VARIANT_PRESETS: dict[str, dict[str, str]] = {
    THINKPRM_VARIANT: {
        "positive_label": "correct",
        "negative_label": "incorrect",
        "scoring_prompt_template": THINKPRM_SCORING_PROMPT_TEMPLATE,
        "decision_prefix": "The step is \\boxed{",
    },
    SUCCESS_PRM_VARIANT: {
        "positive_label": "success",
        "negative_label": "fail",
        "scoring_prompt_template": SUCCESS_PRM_SCORING_PROMPT_TEMPLATE,
        "decision_prefix": "This step is \\boxed{",
    },
}

@dataclass
class LabelTokenCheck:
    """Tokenizer inspection record for one candidate decision label."""
    text: str
    token_ids: list[int]
    token_count: int


@dataclass
class LabelContextResolution:
    """Resolved in-context label forms and their tokenization behavior."""
    positive_text: str
    negative_text: str
    positive_token_ids: list[int]
    negative_token_ids: list[int]
    use_single_token_logits: bool
    mode: str


@dataclass
class ChunkSpan:
    """Token span metadata for one response chunk."""
    start_token_idx: int
    end_token_idx: int
    prefix_before_text: str
    prefix_after_text: str
    chunk_length: int


def _safe_content_to_text(content: Any) -> str:
    """Flatten message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


def raw_prompt_to_text(raw_prompt: Any) -> str:
    """Render a structured chat prompt into a readable text block."""
    if isinstance(raw_prompt, (list, tuple)):
        parts = []
        for message in raw_prompt:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = _safe_content_to_text(message.get("content", ""))
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    return str(raw_prompt)


def compute_valid_prompt_ids(prompts: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the non-padding prompt token ids for one sample."""
    prompt_length = prompts.shape[-1]
    valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
    return prompts[-valid_prompt_length:] if valid_prompt_length > 0 else prompts[:0]


def build_problem_text(
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    raw_prompt: Any | None = None,
) -> str:
    """Build the prompt text that will be shown to the PRM."""
    if raw_prompt is not None:
        return raw_prompt_to_text(raw_prompt)
    valid_prompt_ids = compute_valid_prompt_ids(prompt_ids, attention_mask)
    return tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)




def resolve_prm_variant_config(config) -> tuple[str, str, str, str, str, bool]:
    """Resolve labels and prompt template for the selected PRM variant."""
    variant = str(config.get("prm_variant", THINKPRM_VARIANT)).strip().lower()
    if variant not in SUPPORTED_PRM_VARIANTS:
        raise ValueError(
            f"Unsupported PRM variant: {variant!r}. Expected one of: {sorted(SUPPORTED_PRM_VARIANTS)}"
        )

    default_use_chat_template = True

    if variant == CUSTOM_PRM_VARIANT:
        positive_label = config.get("positive_label")
        negative_label = config.get("negative_label")
        prompt_template = config.get("scoring_prompt_template")
        if not positive_label or not negative_label:
            raise ValueError(
                "algorithm.prm_chunk.prm_variant=custom requires both positive_label and negative_label to be set"
            )
        if not prompt_template:
            prompt_template = DEFAULT_SCORING_PROMPT_TEMPLATE
        decision_prefix = str(config.get("decision_prefix") or DEFAULT_DECISION_PREFIX)
        use_chat_template = bool(config.get("use_chat_template_for_generation", default_use_chat_template))
        return variant, str(positive_label), str(negative_label), str(prompt_template), decision_prefix, use_chat_template

    preset = PRM_VARIANT_PRESETS[variant]
    positive_label = str(config.get("positive_label") or preset["positive_label"])
    negative_label = str(config.get("negative_label") or preset["negative_label"])
    prompt_template = config.get("scoring_prompt_template") or preset["scoring_prompt_template"]
    decision_prefix = str(config.get("decision_prefix") or preset["decision_prefix"])
    use_chat_template = bool(config.get("use_chat_template_for_generation", default_use_chat_template))
    return variant, positive_label, negative_label, str(prompt_template), decision_prefix, use_chat_template


def inspect_label_strings(tokenizer, labels: list[str]) -> list[LabelTokenCheck]:
    """Collect tokenizer diagnostics for candidate decision labels."""
    checks = []
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        checks.append(LabelTokenCheck(text=label, token_ids=token_ids, token_count=len(token_ids)))
    return checks


def _suffix_token_ids(tokenizer, prompt_text: str, label_text: str) -> list[int]:
    """Return only the token ids added by appending a label to a prompt."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + label_text, add_special_tokens=False)
    return full_ids[len(prompt_ids) :]


def resolve_label_context(
    tokenizer,
    prompt_text: str,
    positive_label: str,
    negative_label: str,
) -> LabelContextResolution:
    """Pick the best label rendering in context, preferring single-token labels."""
    candidates = [
        ("plain", positive_label, negative_label),
        ("space_prefixed", f" {positive_label}", f" {negative_label}"),
    ]
    best: Optional[LabelContextResolution] = None
    best_total_len: Optional[int] = None
    for mode, pos_text, neg_text in candidates:
        pos_ids = _suffix_token_ids(tokenizer, prompt_text, pos_text)
        neg_ids = _suffix_token_ids(tokenizer, prompt_text, neg_text)
        use_single = len(pos_ids) == 1 and len(neg_ids) == 1
        total_len = len(pos_ids) + len(neg_ids)
        resolution = LabelContextResolution(
            positive_text=pos_text,
            negative_text=neg_text,
            positive_token_ids=pos_ids,
            negative_token_ids=neg_ids,
            use_single_token_logits=use_single,
            mode=mode,
        )
        if use_single:
            return resolution
        if best is None or total_len < (best_total_len or math.inf):
            best = resolution
            best_total_len = total_len
    assert best is not None
    return best


def build_boundary_prefixes(valid_response_ids: torch.Tensor, tokenizer, boundaries: list[int]) -> list[str]:
    """Decode response prefixes at chunk boundaries for debugging and alignment."""
    prefixes = []
    for boundary in boundaries:
        prefix_ids = valid_response_ids[:boundary]
        prefixes.append(tokenizer.decode(prefix_ids, skip_special_tokens=True))
    return prefixes


def decode_chunk_text(valid_response_ids: torch.Tensor, tokenizer, chunk: ChunkSpan) -> str:
    """Decode the text covered by one chunk span."""
    chunk_ids = valid_response_ids[chunk.start_token_idx : chunk.end_token_idx]
    return tokenizer.decode(chunk_ids, skip_special_tokens=True)


def build_fixed_token_chunks(valid_response_ids: torch.Tensor, tokenizer, chunk_size_tokens: int) -> list[ChunkSpan]:
    """Split a response into near-uniform token-length chunks."""
    valid_length = int(valid_response_ids.numel())
    boundaries = list(range(0, valid_length, chunk_size_tokens))
    if not boundaries or boundaries[-1] != valid_length:
        boundaries.append(valid_length)
    prefixes = build_boundary_prefixes(valid_response_ids, tokenizer, boundaries)
    chunks = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        chunks.append(
            ChunkSpan(
                start_token_idx=start,
                end_token_idx=end,
                prefix_before_text=prefixes[idx],
                prefix_after_text=prefixes[idx + 1],
                chunk_length=end - start,
            )
        )
    return chunks


def _build_step_based_chunks(
    valid_response_ids: torch.Tensor,
    tokenizer,
    fallback_chunk_size_tokens: int,
) -> list[ChunkSpan]:
    """Split on `Step N:` markers, or fall back to fixed-token chunking."""
    response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    matches = list(re.finditer(r"(?im)^step\s+\d+\s*:", response_text))
    if not matches:
        logger.warning("Step-based chunking found no explicit Step N: markers; falling back to fixed token chunking.")
        return build_fixed_token_chunks(valid_response_ids, tokenizer, fallback_chunk_size_tokens)

    step_texts = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(response_text)
        step_texts.append(response_text[start:end])

    lengths = []
    for step_text in step_texts:
        step_ids = tokenizer.encode(step_text, add_special_tokens=False)
        lengths.append(len(step_ids))

    if sum(lengths) != int(valid_response_ids.numel()):
        logger.warning(
            "Step-based chunk tokenization did not round-trip cleanly; falling back to fixed token chunking."
        )
        return build_fixed_token_chunks(valid_response_ids, tokenizer, fallback_chunk_size_tokens)

    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    prefixes = build_boundary_prefixes(valid_response_ids, tokenizer, boundaries)
    return [
        ChunkSpan(
            start_token_idx=boundaries[idx],
            end_token_idx=boundaries[idx + 1],
            prefix_before_text=prefixes[idx],
            prefix_after_text=prefixes[idx + 1],
            chunk_length=boundaries[idx + 1] - boundaries[idx],
        )
        for idx in range(len(boundaries) - 1)
    ]


def build_response_chunks(valid_response_ids: torch.Tensor, tokenizer, config) -> list[ChunkSpan]:
    """Dispatch to the configured chunking strategy."""
    chunking = config.get("chunking", "fixed_token")
    chunk_size_tokens = int(config.get("chunk_size_tokens", 128))
    if chunk_size_tokens <= 0:
        raise ValueError(f"chunk_size_tokens must be positive, got {chunk_size_tokens}")
    if chunking == "fixed_token":
        return build_fixed_token_chunks(valid_response_ids, tokenizer, chunk_size_tokens)
    if chunking == "step_based":
        return _build_step_based_chunks(valid_response_ids, tokenizer, chunk_size_tokens)
    raise ValueError(f"Unsupported PRM chunking mode: {chunking}")


def _strip_step_prefix(text: str) -> str:
    """Remove a leading `Step N:` prefix before re-numbering chunks."""
    return re.sub(r"(?im)^\s*step\s+\d+\s*:\s*", "", text, count=1).strip()


def build_solution_text_from_chunks(valid_response_ids: torch.Tensor, tokenizer, chunks: list[ChunkSpan]) -> str:
    """Normalize chunks into `Step 1: ...` lines for PRM scoring."""
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_text = _strip_step_prefix(decode_chunk_text(valid_response_ids, tokenizer, chunk))
        lines.append(f"Step {idx}: {chunk_text}")
    return "\n".join(lines)


def compute_step_delta_advantages(
    step_scores: list[float],
    final_reward: float,
    use_final_reward_bootstrap: bool,
) -> list[float]:
    """Convert per-step scores into delta-style chunk advantages."""
    if not step_scores:
        return []

    advantages: list[float] = []
    prev_value = 0.0
    for idx, score in enumerate(step_scores):
        value = float(score)
        if use_final_reward_bootstrap and idx == len(step_scores) - 1:
            advantages.append(float(final_reward) - prev_value)
        else:
            advantages.append(value - prev_value)
        prev_value = value
    return advantages


def map_chunk_advantages_to_tokens(
    response_length: int,
    chunks: list[ChunkSpan],
    chunk_advantages: list[float],
    assignment_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast each chunk advantage across the tokens in that chunk."""
    advantages = torch.zeros(response_length, device=device, dtype=dtype)
    if assignment_mode != "broadcast":
        raise ValueError(f"Unsupported chunk_advantage_assignment: {assignment_mode}")
    for chunk, chunk_advantage in zip(chunks, chunk_advantages, strict=True):
        advantages[chunk.start_token_idx : chunk.end_token_idx] = float(chunk_advantage)
    return advantages


def _corrcoef(values_a: list[float], values_b: list[float]) -> float:
    """Compute a safe correlation coefficient for summary metrics."""
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    array_a = np.asarray(values_a, dtype=np.float64)
    array_b = np.asarray(values_b, dtype=np.float64)
    if np.std(array_a) == 0.0 or np.std(array_b) == 0.0:
        return float("nan")
    return float(np.corrcoef(array_a, array_b)[0, 1])


class GenericLabelPRMScorer:
    """Wrap a frozen verifier model and extract per-step correctness scores."""
    def __init__(self, config) -> None:
        """Load the PRM and validate that its decision labels are scoreable."""
        self.config = config
        self.model_path = config.prm_model_path
        self.batch_size = int(config.get("batch_size", 8))
        (
            self.prm_variant,
            self.positive_label,
            self.negative_label,
            self.prompt_template,
            self.decision_prefix,
            self.use_chat_template_for_generation,
        ) = resolve_prm_variant_config(config)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        sample_prompt = self.render_generation_prompt(
            self.build_prompt(problem="1+1=?", solution="Step 1: We add the numbers.")
        )
        label_slot_prefix = (
            f"{sample_prompt}\nLet's verify step by step:\n\n"
            f"Step 1: placeholder critique. {self.decision_prefix}"
        )
        self.label_context = resolve_label_context(
            tokenizer=self.tokenizer,
            prompt_text=label_slot_prefix,
            positive_label=self.positive_label,
            negative_label=self.negative_label,
        )
        self.label_checks = inspect_label_strings(
            self.tokenizer,
            [
                self.positive_label,
                self.negative_label,
                f" {self.positive_label}",
                f" {self.negative_label}",
            ],
        )
        if not self.label_context.use_single_token_logits:
            raise ValueError(
                "One-pass PRM critique scoring requires the positive/negative labels to be single tokens in the scoring context. "
                "Please change the labels or prompt formatting."
            )

    def build_prompt(self, problem: str, solution: str) -> str:
        """Format the problem and normalized solution for the PRM."""
        return self.prompt_template.format(
            problem=problem,
            solution=solution,
            positive_label=self.positive_label,
            negative_label=self.negative_label,
        )

    def render_generation_prompt(self, prompt: str) -> str:
        """Render the verifier prompt through the tokenizer chat template when configured."""
        if not self.use_chat_template_for_generation:
            return prompt
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.warning("Falling back to raw PRM prompt because chat template rendering failed: %s", exc)
            return prompt

    def startup_summary(self) -> str:
        """Return a compact startup summary for debugging tokenizer behavior."""
        label_parts = [
            f"{check.text!r}->{check.token_ids} ({check.token_count} tok)"
            for check in self.label_checks
        ]
        return (
            f"PRM variant={self.prm_variant}; label token check: "
            + ", ".join(label_parts)
            + f"; use_chat_template_for_generation={self.use_chat_template_for_generation}; "
            + f"context_mode={self.label_context.mode}; "
            + f"single_token_logits={self.label_context.use_single_token_logits}; "
            + f"positive_context_ids={self.label_context.positive_token_ids}; "
            + f"negative_context_ids={self.label_context.negative_token_ids}"
        )

    def _prepare_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of prompt strings onto the scorer device."""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _score_single_token_batch(self, prompts: list[str]) -> torch.Tensor:
        """Score prompts from the next-token logits of positive vs negative labels."""
        inputs = self._prepare_inputs(prompts)
        logits = self.model(**inputs).logits
        last_positions = inputs["attention_mask"].sum(dim=-1) - 1
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        next_token_logits = logits[batch_indices, last_positions]
        next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
        pos_token_id = self.label_context.positive_token_ids[0]
        neg_token_id = self.label_context.negative_token_ids[0]
        logp_pos = next_token_log_probs[:, pos_token_id]
        logp_neg = next_token_log_probs[:, neg_token_id]
        stacked = torch.stack([logp_pos, logp_neg], dim=-1)
        return torch.softmax(stacked, dim=-1)[:, 0].detach().cpu()

    def _score_label_sequences(self, prompts: list[str], label_text: str) -> torch.Tensor:
        """Score a multi-token label continuation by summed token logprob."""
        full_texts = [prompt + label_text for prompt in prompts]
        inputs = self._prepare_inputs(full_texts)
        prompt_lengths = torch.tensor(
            [len(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts],
            device=self.device,
            dtype=torch.long,
        )
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        target_ids = inputs["input_ids"][:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        token_positions = torch.arange(token_log_probs.shape[1], device=self.device).unsqueeze(0)
        label_mask = token_positions >= (prompt_lengths.unsqueeze(-1) - 1)
        label_mask = label_mask & (inputs["attention_mask"][:, 1:] > 0)
        return (token_log_probs * label_mask).sum(dim=-1).detach().cpu()

    def _score_sequence_batch(self, prompts: list[str]) -> torch.Tensor:
        """Compare positive and negative multi-token label continuations."""
        logp_pos = self._score_label_sequences(prompts, self.label_context.positive_text)
        logp_neg = self._score_label_sequences(prompts, self.label_context.negative_text)
        stacked = torch.stack([logp_pos, logp_neg], dim=-1)
        return torch.softmax(stacked, dim=-1)[:, 0]

    def _extract_step_scores_from_generation(
        self,
        critique_output: str,
        generated_ids: torch.Tensor,
        generation_scores: tuple[torch.Tensor, ...],
    ) -> list[float]:
        """Read per-step positive-label probabilities from generated critique text."""
        label_pattern = re.compile(rf"\\boxed\{{({re.escape(self.positive_label)}|{re.escape(self.negative_label)})\}}")
        matches = list(label_pattern.finditer(critique_output))
        if not matches:
            logger.warning(
                "PRM critique output did not contain any boxed %s/%s labels.",
                self.positive_label,
                self.negative_label,
            )
            return []

        prefix_to_token_idx = {}
        for token_idx in range(generated_ids.shape[0] + 1):
            prefix_text = self.tokenizer.decode(generated_ids[:token_idx], skip_special_tokens=True)
            prefix_to_token_idx[prefix_text] = token_idx

        pos_token_id = self.label_context.positive_token_ids[0]
        neg_token_id = self.label_context.negative_token_ids[0]
        step_scores: list[float] = []
        for match in matches:
            prefix_text = critique_output[: match.start(1)]
            token_idx = prefix_to_token_idx.get(prefix_text)
            if token_idx is None or token_idx >= len(generation_scores):
                logger.warning("Could not align PRM label position with generation scores; skipping one step score.")
                continue
            token_logits = generation_scores[token_idx][0]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            stacked = torch.stack([log_probs[pos_token_id], log_probs[neg_token_id]], dim=-1)
            step_scores.append(torch.softmax(stacked, dim=-1)[0].item())
        return step_scores

    @torch.no_grad()
    def generate_step_critique(self, problem: str, solution: str) -> tuple[str, list[float]]:
        """Generate a critique and convert its boxed decisions into step scores."""
        prompt = self.build_prompt(problem=problem, solution=solution)
        rendered_prompt = self.render_generation_prompt(prompt)
        inputs = self._prepare_inputs([rendered_prompt])
        max_new_tokens = int(self.config.get("max_new_tokens", 512))
        generation = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = generation.sequences[0, prompt_len:]
        critique_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        step_scores = self._extract_step_scores_from_generation(
            critique_output=critique_output,
            generated_ids=generated_ids,
            generation_scores=generation.scores,
        )
        return critique_output, step_scores



class PRMChunkAdvantageEstimator:
    """Compute PPO advantages from PRM scores on response chunks."""
    def __init__(self, rollout_tokenizer, config) -> None:
        """Create the scorer used during PPO advantage computation."""
        self.rollout_tokenizer = rollout_tokenizer
        self.config = config
        self.scorer = GenericLabelPRMScorer(config)
        self.last_startup_summary = self.scorer.startup_summary()

    def compute(self, batch: DataProto) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Score each sample chunk-by-chunk and return token-level advantages."""
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        prompts = batch.batch["prompts"]
        token_level_scores = batch.batch["token_level_scores"]

        batch_size, response_length = responses.shape
        advantages = torch.zeros_like(responses, dtype=torch.float32)
        returns = torch.zeros_like(responses, dtype=torch.float32)

        all_chunk_lengths: list[float] = []
        all_chunk_scores: list[float] = []
        all_chunk_advantages: list[float] = []
        chunk_counts: list[float] = []
        final_rewards: list[float] = []
        scored_step_counts: list[float] = []

        for idx in range(batch_size):
            valid_length = int(response_mask[idx].sum().item())
            valid_response_ids = responses[idx, :valid_length]
            raw_prompt = batch.non_tensor_batch.get("raw_prompt", None)
            raw_prompt_item = raw_prompt[idx] if raw_prompt is not None else None
            problem_text = build_problem_text(
                tokenizer=self.rollout_tokenizer,
                prompt_ids=prompts[idx],
                attention_mask=attention_mask[idx],
                raw_prompt=raw_prompt_item,
            )
            chunks = build_response_chunks(valid_response_ids, self.rollout_tokenizer, self.config)
            solution_text = build_solution_text_from_chunks(valid_response_ids, self.rollout_tokenizer, chunks)
            critique_output, chunk_scores = self.scorer.generate_step_critique(
                problem=problem_text,
                solution=solution_text,
            )
            if len(chunk_scores) < len(chunks):
                logger.warning(
                    "PRM critique produced %s step scores for %s chunks. Padding missing scores with zeros.",
                    len(chunk_scores),
                    len(chunks),
                )
                chunk_scores.extend([0.0] * (len(chunks) - len(chunk_scores)))
            if len(chunk_scores) > len(chunks):
                chunk_scores = chunk_scores[: len(chunks)]

            final_reward = float(token_level_scores[idx].sum().item())
            final_rewards.append(final_reward)
            scored_step_counts.append(float(len(chunk_scores)))
            chunk_advantages = compute_step_delta_advantages(
                step_scores=chunk_scores,
                final_reward=final_reward,
                use_final_reward_bootstrap=bool(self.config.get("use_final_reward_bootstrap", False)),
            )

            for chunk, chunk_score, chunk_advantage in zip(chunks, chunk_scores, chunk_advantages, strict=True):
                all_chunk_lengths.append(float(chunk.chunk_length))
                all_chunk_scores.append(float(chunk_score))
                all_chunk_advantages.append(float(chunk_advantage))

            token_advantages = map_chunk_advantages_to_tokens(
                response_length=response_length,
                chunks=chunks,
                chunk_advantages=chunk_advantages,
                assignment_mode=self.config.get("chunk_advantage_assignment", "broadcast"),
                device=responses.device,
                dtype=torch.float32,
            )
            token_advantages = token_advantages * response_mask[idx].to(token_advantages.dtype)
            advantages[idx] = token_advantages
            returns[idx] = token_advantages
            chunk_counts.append(float(len(chunks)))

        metrics = {
            "prm_chunk/chunk_score_mean": float(np.mean(all_chunk_scores)) if all_chunk_scores else float("nan"),
            "prm_chunk/chunk_score_std": float(np.std(all_chunk_scores)) if all_chunk_scores else float("nan"),
            "prm_chunk/chunk_abs_score_mean": (
                float(np.mean(np.abs(all_chunk_scores))) if all_chunk_scores else float("nan")
            ),
            "prm_chunk/chunk_adv_mean": float(np.mean(all_chunk_advantages)) if all_chunk_advantages else float("nan"),
            "prm_chunk/chunk_adv_std": float(np.std(all_chunk_advantages)) if all_chunk_advantages else float("nan"),
            "prm_chunk/chunk_abs_adv_mean": (
                float(np.mean(np.abs(all_chunk_advantages))) if all_chunk_advantages else float("nan")
            ),
            "prm_chunk/chunk_length_mean": float(np.mean(all_chunk_lengths)) if all_chunk_lengths else float("nan"),
            "prm_chunk/chunk_length_score_corr": _corrcoef(all_chunk_lengths, all_chunk_scores),
            "prm_chunk/chunk_length_abs_score_corr": (
                _corrcoef(all_chunk_lengths, [abs(value) for value in all_chunk_scores])
            ),
            "prm_chunk/chunk_length_adv_corr": _corrcoef(all_chunk_lengths, all_chunk_advantages),
            "prm_chunk/chunk_length_abs_adv_corr": (
                _corrcoef(all_chunk_lengths, [abs(value) for value in all_chunk_advantages])
            ),
            "prm_chunk/final_reward_mean": float(np.mean(final_rewards)) if final_rewards else float("nan"),
            "prm_chunk/chunks_per_sample_mean": float(np.mean(chunk_counts)) if chunk_counts else float("nan"),
            "prm_chunk/scored_steps_per_sample_mean": (
                float(np.mean(scored_step_counts)) if scored_step_counts else float("nan")
            ),
            "prm_chunk/score_gt_half_pct": (
                float(np.mean([value > 0.5 for value in all_chunk_scores])) if all_chunk_scores else float("nan")
            ),
        }
        return advantages, returns, metrics
