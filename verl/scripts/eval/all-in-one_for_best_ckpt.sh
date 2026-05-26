#!/usr/bin/env bash

cd $ROOT/verl/recipe/eval

HOME="$HOME"

BEST_CKPTS=(
  "GRPO_Qwen3-8B-Base_ours_dapo_17k_v2_deepmath_16384_temp-1.0:450"
)


for pair in "${BEST_CKPTS[@]}"; do
  EXP_NAME="${pair%%:*}"
  STEP="${pair##*:}"
  skip=0
  for s in "${SKIP_EXPS[@]}"; do
    if [[ "$EXP_NAME" == "$s" ]]; then skip=1; break; fi
  done
  if [[ "$skip" -eq 1 ]]; then
    echo "Skip (excluded): $EXP_NAME"
    continue
  fi
  OUTPUT_PATH="./results/outputs/${EXP_NAME}_${STEP}"
  if [[ -f "$OUTPUT_PATH" ]] || [[ -d "$OUTPUT_PATH" ]]; then
    echo "Skip (exists): $EXP_NAME @ step $STEP"
    continue
  fi
  echo "Running: $EXP_NAME @ global_step_${STEP}"
  LOCAL_DIR=$HOME/checkpoints/ATLAS/${EXP_NAME}/global_step_${STEP}/actor
  TARGET_DIR=$HOME/checkpoints/ATLAS/${EXP_NAME}/global_step_${STEP}/actor/hf
  python -m verl.model_merger merge --backend fsdp --local_dir "$LOCAL_DIR" --target_dir "$TARGET_DIR"
  MODEL_PATH=$HOME/checkpoints/ATLAS/${EXP_NAME}/global_step_${STEP}/actor/hf
  CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
  python3 ./eval_light.py \
    --data_dir ./data \
    --output_dir ./results/outputs/${EXP_NAME}_${STEP} \
    --completions_save_dir ./results/completions/${EXP_NAME}_${STEP} \
    --model_name_or_path "$MODEL_PATH" \
    --result_filename "${EXP_NAME}-${STEP}.json" \
    --temperature 0.6 \
    --max_tokens 16384 \
    --seed 0 \
    --top_p 0.95 \
    --top_k 20
done

