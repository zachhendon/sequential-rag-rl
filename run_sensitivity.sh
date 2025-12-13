#!/bin/bash
# Sensitivity Analysis: How sensitive is performance to hyperparameters?
# Using Qwen2.5-0.5B for faster experimentation
# Usage: ./run_sensitivity.sh [0.5B|1.5B] [all|learning_rate|num_examples|hidden_size|batch_size|entropy|gamma]

# Model selection: Use command line argument or default to 0.5B
if [ -z "$1" ]; then
    MODEL_SIZE="0.5B"
else
    MODEL_SIZE="$1"
fi

# Which experiments to run
EXPERIMENT="${2:-all}"

# Model settings - adjust based on model size
if [ "$MODEL_SIZE" = "1.5B" ]; then
    GENERATOR_MODEL="Qwen/Qwen2.5-1.5B"
    MAX_GEN_TOKENS=256
    BASE_BATCH_SIZE=24      # Slightly smaller batch for larger model
    GPU_MEM_UTIL=0.3        # Reduced for sensitivity experiments
    echo "Using Qwen2.5-1.5B model"
elif [ "$MODEL_SIZE" = "0.5B" ]; then
    GENERATOR_MODEL="Qwen/Qwen2.5-0.5B"
    MAX_GEN_TOKENS=256
    BASE_BATCH_SIZE=32      # Can use larger batch with smaller model
    GPU_MEM_UTIL=0.3        # Lower GPU memory utilization for 0.5B
    echo "Using Qwen2.5-0.5B model"
else
    echo "Error: Invalid model size '$MODEL_SIZE'. Use '0.5B' or '1.5B'"
    exit 1
fi

# Base settings
DATASET="gsm8k"
EPOCHS=20
TRAIN_SIZE=1000
VAL_SIZE=200
BASE_LR=1e-4
BASE_HIDDEN_SIZE=512
BASE_NUM_EXAMPLES=4

mkdir -p results/sensitivity

echo "=========================================="
echo "Running Sensitivity Analysis"
echo "Generator Model: ${GENERATOR_MODEL}"
echo "Model Size: ${MODEL_SIZE}"
echo "Experiment: ${EXPERIMENT}"
echo "=========================================="

# Sensitivity 1: Number of Retrieved Examples (k)
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "num_examples" ]; then
    echo "[Sensitivity 1] Number of Examples (k)..."
    for K in 1 2 3 4 5 6; do
        echo "  Training with k=${K} examples..."
        python run.py --train \
            --dataset $DATASET \
            --rl_algo ppo \
            --model_type lstm \
            --model_name results/sensitivity/${MODEL_SIZE}_k_${K} \
            --num_examples $K \
            --epochs $EPOCHS \
            --batch_size $BASE_BATCH_SIZE \
            --lr $BASE_LR \
            --hidden_size $BASE_HIDDEN_SIZE \
            --generator_model $GENERATOR_MODEL \
            --max_gen_tokens $MAX_GEN_TOKENS \
            --gpu_memory_utilization $GPU_MEM_UTIL \
            --train_size $TRAIN_SIZE \
            --val_size $VAL_SIZE \
            --wandb \
            2>&1 | tee results/sensitivity/${MODEL_SIZE}_k_${K}.log
    done
fi

# Sensitivity 2: Learning Rate
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "learning_rate" ]; then
    echo "[Sensitivity 2] Learning Rate..."
    for LR in 1e-5 5e-5 1e-4 5e-4 1e-3; do
        LR_STR=$(echo $LR | sed 's/\./_/g' | sed 's/-/_/g')
        echo "  Training with lr=${LR}..."
        python run.py --train \
            --dataset $DATASET \
            --rl_algo ppo \
            --model_type lstm \
            --model_name results/sensitivity/${MODEL_SIZE}_lr_${LR_STR} \
            --num_examples $BASE_NUM_EXAMPLES \
            --epochs $EPOCHS \
            --batch_size $BASE_BATCH_SIZE \
            --lr $LR \
            --hidden_size $BASE_HIDDEN_SIZE \
            --generator_model $GENERATOR_MODEL \
            --max_gen_tokens $MAX_GEN_TOKENS \
            --gpu_memory_utilization $GPU_MEM_UTIL \
            --train_size $TRAIN_SIZE \
            --val_size $VAL_SIZE \
            --wandb \
            2>&1 | tee results/sensitivity/${MODEL_SIZE}_lr_${LR_STR}.log
    done
fi

# Sensitivity 3: Hidden Size
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "hidden_size" ]; then
    echo "[Sensitivity 3] Hidden Size..."
    for HIDDEN in 256 512 800; do
        echo "  Training with hidden_size=${HIDDEN}..."
        python run.py --train \
            --dataset $DATASET \
            --rl_algo ppo \
            --model_type lstm \
            --model_name results/sensitivity/${MODEL_SIZE}_hidden_${HIDDEN} \
            --num_examples $BASE_NUM_EXAMPLES \
            --epochs $EPOCHS \
            --batch_size $BASE_BATCH_SIZE \
            --lr $BASE_LR \
            --hidden_size $HIDDEN \
            --generator_model $GENERATOR_MODEL \
            --max_gen_tokens $MAX_GEN_TOKENS \
            --gpu_memory_utilization $GPU_MEM_UTIL \
            --train_size $TRAIN_SIZE \
            --val_size $VAL_SIZE \
            --wandb \
            2>&1 | tee results/sensitivity/${MODEL_SIZE}_hidden_${HIDDEN}.log
    done
fi

# Sensitivity 4: Batch Size
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "batch_size" ]; then
    echo "[Sensitivity 4] Batch Size..."
    # Adjust batch sizes based on model size
    if [ "$MODEL_SIZE" = "1.5B" ]; then
        BATCH_SIZES=(16 20 24 28)
    else
        BATCH_SIZES=(16 24 32 40)
    fi
    for BS in "${BATCH_SIZES[@]}"; do
        echo "  Training with batch_size=${BS}..."
        python run.py --train \
            --dataset $DATASET \
            --rl_algo ppo \
            --model_type lstm \
            --model_name results/sensitivity/${MODEL_SIZE}_batch_${BS} \
            --num_examples $BASE_NUM_EXAMPLES \
            --epochs $EPOCHS \
            --batch_size $BS \
            --lr $BASE_LR \
            --hidden_size $BASE_HIDDEN_SIZE \
            --generator_model $GENERATOR_MODEL \
            --max_gen_tokens $MAX_GEN_TOKENS \
            --gpu_memory_utilization $GPU_MEM_UTIL \
            --train_size $TRAIN_SIZE \
            --val_size $VAL_SIZE \
            --wandb \
            2>&1 | tee results/sensitivity/${MODEL_SIZE}_batch_${BS}.log
    done
fi

# Sensitivity 5: Entropy Coefficient (exploration)
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "entropy" ]; then
    echo "[Sensitivity 5] Entropy Coefficient..."
    for E_COEF in 0.0 0.01 0.05 0.1; do
        E_COEF_STR=$(echo $E_COEF | sed 's/\./_/g')
        echo "  Training with e_coef=${E_COEF}..."
        python run.py --train \
            --dataset $DATASET \
            --rl_algo ppo \
            --model_type lstm \
            --model_name results/sensitivity/${MODEL_SIZE}_entropy_${E_COEF_STR} \
            --num_examples $BASE_NUM_EXAMPLES \
            --epochs $EPOCHS \
            --batch_size $BASE_BATCH_SIZE \
            --lr $BASE_LR \
            --hidden_size $BASE_HIDDEN_SIZE \
            --e_coef $E_COEF \
            --generator_model $GENERATOR_MODEL \
            --max_gen_tokens $MAX_GEN_TOKENS \
            --gpu_memory_utilization $GPU_MEM_UTIL \
            --train_size $TRAIN_SIZE \
            --val_size $VAL_SIZE \
            --wandb \
            2>&1 | tee results/sensitivity/${MODEL_SIZE}_entropy_${E_COEF_STR}.log
    done
fi

# Sensitivity 6: Discount Factor (gamma)
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "gamma" ]; then
    echo "[Sensitivity 6] Discount Factor (gamma)..."
    for GAMMA in 0.9 0.95 0.99 1.0; do
        GAMMA_STR=$(echo $GAMMA | sed 's/\./_/g')
        echo "  Training with gamma=${GAMMA}..."
        python run.py --train \
            --dataset $DATASET \
            --rl_algo ppo \
            --model_type lstm \
            --model_name results/sensitivity/${MODEL_SIZE}_gamma_${GAMMA_STR} \
            --num_examples $BASE_NUM_EXAMPLES \
            --epochs $EPOCHS \
            --batch_size $BASE_BATCH_SIZE \
            --lr $BASE_LR \
            --hidden_size $BASE_HIDDEN_SIZE \
            --gamma $GAMMA \
            --generator_model $GENERATOR_MODEL \
            --max_gen_tokens $MAX_GEN_TOKENS \
            --gpu_memory_utilization $GPU_MEM_UTIL \
            --train_size $TRAIN_SIZE \
            --val_size $VAL_SIZE \
            --wandb \
            2>&1 | tee results/sensitivity/${MODEL_SIZE}_gamma_${GAMMA_STR}.log
    done
fi

echo "=========================================="
echo "Sensitivity Analysis Complete!"
echo "Model: ${MODEL_SIZE}"
echo "Results saved to results/sensitivity/"
echo "=========================================="
echo ""
echo "To analyze results, run:"
echo "  python analyze_sensitivity.py ${MODEL_SIZE}"