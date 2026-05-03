set -euo pipefail

PRM_VARIANT=${PRM_VARIANT:-thinkprm}

python3 -m verl.trainer.main_ppo \
  --config-name=ppo_trainer \
  algorithm.adv_estimator=prm_chunk \
  algorithm.prm_chunk.enable=true \
  algorithm.prm_chunk.prm_model_path=path/to/prm \
  algorithm.prm_chunk.prm_variant=${PRM_VARIANT} \
  algorithm.prm_chunk.chunking=step_based \
  algorithm.prm_chunk.chunk_size_tokens=128 \
  algorithm.prm_chunk.chunk_advantage_assignment=broadcast \
  algorithm.prm_chunk.advantage_mode=delta_value \
  algorithm.prm_chunk.use_final_reward_bootstrap=false \
  algorithm.prm_chunk.log_chunk_length_bias=true
