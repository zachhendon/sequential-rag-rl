#!/bin/bash
# Main Experiment: Compare sequential RL retrieval vs baselines
# Supports both Qwen2.5-0.5B and Qwen2.5-1.5B models
# Research Question: Does sequential RL retrieval improve RAG performance over static methods?

# Model selection: Use command line argument or default to 0.5B
# Usage: ./run_main_experiment_fast.sh [0.5B|1.5B]
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

# Common settings
DATASET="gsm8k"
NUM_EXAMPLES=4
EPOCHS=20          # Reduced epochs for faster iteration
LR=1e-4
HIDDEN_SIZE=512    # Slightly smaller hidden size

# Create results directory
mkdir -p results/main_experiment

echo "=========================================="
echo "Running Main Experiment on ${DATASET}"
echo "Generator Model: ${GENERATOR_MODEL}"
echo "Model Size: ${MODEL_SIZE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU Memory Utilization: ${GPU_MEM_UTIL}"
echo "=========================================="

# Baseline 1: Random retrieval
echo "[1/6] Running Random Baseline..."
python run.py --eval test \
    --dataset $DATASET \
    --sm random \
    --num_examples $NUM_EXAMPLES \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --val_size 200 \
    2>&1 | tee results/main_experiment/random_baseline_${MODEL_SIZE}.log

# Baseline 2: kNN (similarity-based) retrieval
echo "[2/6] Running kNN Baseline..."
python run.py --eval test \
    --dataset $DATASET \
    --sm sim \
    --num_examples $NUM_EXAMPLES \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --val_size 200 \
    2>&1 | tee results/main_experiment/knn_baseline_${MODEL_SIZE}.log

# Method 1: REINFORCE with baseline
echo "[3/6] Training REINFORCE with Baseline..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo rwb \
    --model_type lstm \
    --model_name results/main_experiment/rwb_lstm_${MODEL_SIZE} \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --train_size 1000 \
    --val_size 200 \
    --wandb \
    2>&1 | tee results/main_experiment/rwb_train_${MODEL_SIZE}.log

# Method 2: Actor-Critic
echo "[4/6] Training Actor-Critic..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo ac \
    --model_type lstm \
    --model_name results/main_experiment/ac_lstm_${MODEL_SIZE} \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --train_size 1000 \
    --val_size 200 \
    --wandb \
    2>&1 | tee results/main_experiment/ac_train_${MODEL_SIZE}.log

# Method 3: PPO
echo "[5/6] Training PPO..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo ppo \
    --model_type lstm \
    --model_name results/main_experiment/ppo_lstm_${MODEL_SIZE} \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --e_coef 0.01 \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --train_size 1000 \
    --val_size 200 \
    --wandb \
    2>&1 | tee results/main_experiment/ppo_train_${MODEL_SIZE}.log

# Method 4: Independent (non-sequential) model with PPO
echo "[6/6] Training Independent Model (Non-sequential baseline) ..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo ppo \
    --model_type ind \
    --model_name results/main_experiment/ppo_ind_${MODEL_SIZE} \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_size $HIDDEN_SIZE \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --train_size 1000 \
    --val_size 200 \
    --wandb \
    2>&1 | tee results/main_experiment/ppo_ind_train_${MODEL_SIZE}.log

echo "=========================================="
echo "Main Experiment Complete!"
echo "Model: ${GENERATOR_MODEL} (${MODEL_SIZE})"
echo "Results saved to results/main_experiment/"
echo "=========================================="