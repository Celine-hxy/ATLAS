#!/usr/bin/env bash
# Effective-data eval at each run's best global_step (same BEST_CKPTS as all-in-one_for_best_ckpt.sh).

cd $ROOT/verl/recipe/eval

HOME="$HOME"
RESULT_ROOT="$ROOT/verl/recipe/eval/results_effective_data_best"


BEST_CKPTS=(
  "GRPO_Qwen3-1.7B-Base_cn_k12_16384_temp-1.0_wo_mcq:350"
  "GRPO_Qwen3-1.7B-Base_olympiads_16384_temp-1.0_wo_mcq:200"
  "GRPO_Qwen3-1.7B-Base_amc_aime_16384_temp-1.0_wo_mcq:500"
  "GRPO_Qwen3-1.7B-Base_dapo_16384_temp-1.0:50"
  "GRPO_Qwen3-1.7B-Base_gsm8k_16384_temp-1.0:100"
  "GRPO_Qwen3-1.7B-Base_olympiads_16384_temp-1.0:50"
  "GRPO_Qwen3-1.7B-Base_numina_math1.5_16384_temp-1.0:200"
  "GRPO_Qwen3-1.7B-Base_aops_forum_16384_temp-1.0:50"
  "GRPO_Qwen3-1.7B-Base_basic_arithmetic_16384_temp-1.0:200"
  "GRPO_Qwen3-1.7B-Base_synthetic_amc_16384_temp-1.0:150"
  "GRPO_Qwen3-1.7B-Base_cn_k12_16384_temp-1.0_re:250"
  "GRPO_Qwen3-1.7B-Base_lila_synthetic_16384_temp-1.0:200"
  "GRPO_Qwen3-1.7B-Base_areal_boba_16384_temp-1.0:200"
  "GRPO_Qwen3-1.7B-Base_metamath_16384_temp-1.0:200"
  "GRPO_Qwen3-1.7B-Base_still1_16384_temp-1.0:200"
  "GRPO_Qwen3-1.7B-Base_num_glue_16384_temp-1.0:450"
  "GRPO_Qwen3-1.7B-Base_stack_exchange_16384_temp-1.0:100"
  "GRPO_Qwen3-1.7B-Base_amc_aime_16384_temp-1.0__:300"
  "GRPO_Qwen3-1.7B-Base_lila_crawl_16384_temp-1.0:250"
  "GRPO_Qwen3-1.7B-Base_math_16384_temp-1.0__:250"
  "GRPO_Qwen3-1.7B-Base_big_math_16384_temp-1.0_re:100"
  "GRPO_Qwen3-1.7B-Base__orca_math_16384_temp-1.0:250"
  "GRPO_Qwen3-1.7B-Base_synthetic_math_16384_temp-1.0:400"
  "GRPO_Qwen3-1.7B-Base_test_leak_16384_temp-1.0:150"
  "GRPO_Qwen3-1.7B-Base_still3_16384_temp-1.0:500"
)

for pair in "${BEST_CKPTS[@]}"; do
  EXP_NAME="${pair%%:*}"
  STEP="${pair##*:}"
  skip=0
  if [[ "$skip" -eq 1 ]]; then
    echo "Skip (excluded): $EXP_NAME"
    continue
  fi
  OUT_TAG="${EXP_NAME}_${STEP}"
  OUT="${RESULT_ROOT}/outputs/${OUT_TAG}"
  if [[ -d "$OUT" ]]; then
    echo "Skip (exists): $EXP_NAME @ step $STEP"
    continue
  fi
  echo "Running: $EXP_NAME @ global_step_${STEP}"
  LOCAL_DIR=$HOME/checkpoints/ATLAS/${EXP_NAME}/global_step_${STEP}/actor
  TARGET_DIR=$HOME/checkpoints/ATLAS/${EXP_NAME}/global_step_${STEP}/actor/hf
  (cd $ROOT/verl && python -m verl.model_merger merge --backend fsdp --local_dir "$LOCAL_DIR" --target_dir "$TARGET_DIR")
  CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
  python3 ./eval_effective_data.py \
    --eval_data_source train_parquet \
    --ancestor_data_root "${HOME}/data/ancestor_data" \
    --output_dir "${RESULT_ROOT}/outputs/${OUT_TAG}" \
    --completions_save_dir "${RESULT_ROOT}/completions/${OUT_TAG}" \
    --metrics_dir "${RESULT_ROOT}" \
    --checkpoint_home "$HOME" \
    --exp_name "$EXP_NAME" \
    --global_step "$STEP" \
    --result_filename "${EXP_NAME}-${STEP}.json" \
    --temperature 0.6 \
    --max_tokens 16384 \
    --seed 0 \
    --top_p 0.95 \
    --top_k 20
done
