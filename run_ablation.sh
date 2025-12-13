#!/bin/bash
# Ablation Study: Impact of different components
# Supports both Qwen2.5-0.5B and Qwen2.5-1.5B models
# Usage: ./run_ablation.sh [0.5B|1.5B]

# Model selection: Use command line argument or default to 0.5B
if [ -z "$1" ]; then
    MODEL_SIZE="0.5B"
else
    MODEL_SIZE="$1"
fi

# Model settings - adjust based on model size
if [ "$MODEL_SIZE" = "1.5B" ]; then
    GENERATOR_MODEL="Qwen/Qwen2.5-1.5B"
    MAX_GEN_TOKENS=256
    BATCH_SIZE=24      # Slightly smaller batch for larger model
    GPU_MEM_UTIL=0.4   # Higher GPU memory utilization for 1.5B
    echo "Using Qwen2.5-1.5B model"
elif [ "$MODEL_SIZE" = "0.5B" ]; then
    GENERATOR_MODEL="Qwen/Qwen2.5-0.5B"
    MAX_GEN_TOKENS=256
    BATCH_SIZE=32      # Can use larger batch with smaller model
    GPU_MEM_UTIL=0.3   # Lower GPU memory utilization for 0.5B
    echo "Using Qwen2.5-0.5B model"
else
    echo "Error: Invalid model size '$MODEL_SIZE'. Use '0.5B' or '1.5B'"
    exit 1
fi

# Base settings
DATASET="gsm8k"
NUM_EXAMPLES=4
EPOCHS=20
LR=1e-4
HIDDEN_SIZE=512
RL_ALGO="ppo"
TRAIN_SIZE=1000
VAL_SIZE=200

mkdir -p results/ablation

echo "=========================================="
echo "Running Ablation Studies"
echo "Generator Model: ${GENERATOR_MODEL}"
echo "Model Size: ${MODEL_SIZE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU Memory Utilization: ${GPU_MEM_UTIL}"
echo "=========================================="

# Ablation 1: Model Architecture
echo "[Ablation 1] Model Architecture Comparison..."
for MODEL_TYPE in lstm rnn attn ind; do
    echo "  Training ${MODEL_TYPE}..."
    python run.py --train \
        --dataset $DATASET \
        --rl_algo $RL_ALGO \
        --model_type $MODEL_TYPE \
        --model_name results/ablation/${MODEL_SIZE}_arch_${MODEL_TYPE} \
        --num_examples $NUM_EXAMPLES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --hidden_size $HIDDEN_SIZE \
        --generator_model $GENERATOR_MODEL \
        --max_gen_tokens $MAX_GEN_TOKENS \
        --gpu_memory_utilization $GPU_MEM_UTIL \
        --train_size $TRAIN_SIZE \
        --val_size $VAL_SIZE \
        --wandb \
        2>&1 | tee results/ablation/${MODEL_SIZE}_arch_${MODEL_TYPE}.log
done

# Ablation 2: Intermediate Rewards
echo "[Ablation 2] Intermediate Reward Comparison..."

# No intermediate reward (sparse only)
echo "  Training without intermediate rewards..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo $RL_ALGO \
    --model_type lstm \
        --model_name results/ablation/${MODEL_SIZE}_reward_sparse \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --int_reward_sim 0 \
    --int_reward_multi 0 \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --wandb \
        2>&1 | tee results/ablation/${MODEL_SIZE}_reward_sparse.log

# Similarity-based intermediate reward
echo "  Training with similarity-based intermediate rewards..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo $RL_ALGO \
    --model_type lstm \
        --model_name results/ablation/${MODEL_SIZE}_reward_sim \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --int_reward_sim 1 \
    --int_reward_multi 0 \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --wandb \
        2>&1 | tee results/ablation/${MODEL_SIZE}_reward_sim.log

# Ablation 3: Early Stopping (Dynamic # of documents)
echo "[Ablation 3] Early Stopping Comparison..."

# Without early stopping (fixed k documents)
echo "  Training without early stopping..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo $RL_ALGO \
    --model_type lstm \
        --model_name results/ablation/${MODEL_SIZE}_no_early_stop \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --early_stopping 0 \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --wandb \
        2>&1 | tee results/ablation/${MODEL_SIZE}_no_early_stop.log

# With early stopping
echo "  Training with early stopping..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo $RL_ALGO \
    --model_type lstm \
        --model_name results/ablation/${MODEL_SIZE}_early_stop \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --early_stopping 1 \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --wandb \
        2>&1 | tee results/ablation/${MODEL_SIZE}_early_stop.log

# Ablation 4: RL Algorithm Comparison
echo "[Ablation 4] RL Algorithm Comparison..."
for ALGO in reinforce rwb ac ppo; do
    echo "  Training ${ALGO}..."
    python run.py --train \
        --dataset $DATASET \
        --rl_algo $ALGO \
        --model_type lstm \
        --model_name results/ablation/${MODEL_SIZE}_algo_${ALGO} \
        --num_examples $NUM_EXAMPLES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --hidden_size $HIDDEN_SIZE \
        --generator_model $GENERATOR_MODEL \
        --max_gen_tokens $MAX_GEN_TOKENS \
        --gpu_memory_utilization $GPU_MEM_UTIL \
        --train_size $TRAIN_SIZE \
        --val_size $VAL_SIZE \
        --wandb \
        2>&1 | tee results/ablation/${MODEL_SIZE}_algo_${ALGO}.log
done

echo "=========================================="
echo "Ablation Studies Complete!"
echo "Model: ${MODEL_SIZE}"
echo "Results saved to results/ablation/"
echo "=========================================="