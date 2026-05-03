from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.trainer.ppo.prm_chunk import (
    GenericLabelPRMScorer,
    build_fixed_token_chunks,
    build_solution_text_from_chunks,
    map_chunk_advantages_to_tokens,
    resolve_label_context,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prm-variant", default="thinkprm", choices=["thinkprm", "success_prm", "custom"])
    parser.add_argument("--positive-label")
    parser.add_argument("--negative-label")
    parser.add_argument("--chunk-size-tokens", type=int, default=8)
    parser.add_argument("--problem", default="What is 2 + 2?")
    parser.add_argument(
        "--solution",
        default="Step 1: We add 2 and 2.\nStep 2: The result should be 4.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "prm_model_path": args.model_path,
            "prm_variant": args.prm_variant,
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "scoring_prompt_template": None,
            "decision_prefix": None,
            "batch_size": 4,
            "max_new_tokens": 256,
        }
    )
    scorer = GenericLabelPRMScorer(config)
    print(scorer.startup_summary())

    labels = [
        scorer.positive_label,
        scorer.negative_label,
        f" {scorer.positive_label}",
        f" {scorer.negative_label}",
    ]
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        print(f"label={label!r} token_ids={token_ids} token_count={len(token_ids)}")

    prompt = scorer.build_prompt(problem=args.problem, solution=args.solution)
    label_slot_prefix = (
        f"{prompt}\n\nLet's verify step by step:\n\n"
        "Step 1: placeholder critique. This step is \\boxed{{"
    )
    resolution = resolve_label_context(
        tokenizer,
        label_slot_prefix,
        scorer.positive_label,
        scorer.negative_label,
    )
    print(
        f"context_mode={resolution.mode} single_token_logits={resolution.use_single_token_logits} "
        f"positive_context_ids={resolution.positive_token_ids} negative_context_ids={resolution.negative_token_ids}"
    )

    response_ids = tokenizer.encode(args.solution, add_special_tokens=False)
    response_tensor = torch.tensor(response_ids, dtype=torch.long)
    chunks = build_fixed_token_chunks(response_tensor, tokenizer, args.chunk_size_tokens)
    solution_text = build_solution_text_from_chunks(response_tensor, tokenizer, chunks)
    print("chunk_spans=", [(chunk.start_token_idx, chunk.end_token_idx, chunk.chunk_length) for chunk in chunks])
    print("normalized_solution=")
    print(solution_text)

    critique_output, step_scores = scorer.generate_step_critique(problem=args.problem, solution=solution_text)
    print("critique_output=")
    print(critique_output)
    print(f"step_scores={step_scores}")

    if len(step_scores) < len(chunks):
        step_scores.extend([0.0] * (len(chunks) - len(step_scores)))
    step_scores = step_scores[: len(chunks)]

    token_advantages = map_chunk_advantages_to_tokens(
        response_length=len(response_ids),
        chunks=chunks,
        chunk_advantages=step_scores,
        assignment_mode="broadcast",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    print(f"token_advantages_shape={tuple(token_advantages.shape)}")
    print(f"token_advantages={token_advantages.tolist()}")


if __name__ == "__main__":
    main()
