#!/bin/bash
# Quick Test Script: Verify everything works before full experiments
# This runs a minimal experiment to check setup

# Model settings - smallest available
GENERATOR_MODEL="Qwen/Qwen2.5-0.5B"
MAX_GEN_TOKENS=128

# Minimal settings for quick test
DATASET="gsm8k"
NUM_EXAMPLES=2
EPOCHS=2
BATCH_SIZE=8
TRAIN_SIZE=50
VAL_SIZE=20

mkdir -p results/test

echo "=========================================="
echo "Quick Test: Verifying Setup"
echo "Generator Model: ${GENERATOR_MODEL}"
echo "This should complete in ~5-10 minutes"
echo "=========================================="

# Test 1: Random baseline evaluation
echo "[Test 1/3] Testing Random Baseline Evaluation..."
python run.py --eval test \
    --dataset $DATASET \
    --sm random \
    --num_examples $NUM_EXAMPLES \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --val_size $VAL_SIZE \
    2>&1 | tee results/test/random_test.log

if [ $? -eq 0 ]; then
    echo "✓ Random baseline test PASSED"
else
    echo "✗ Random baseline test FAILED"
    exit 1
fi

# Test 2: kNN baseline evaluation
echo "[Test 2/3] Testing kNN Baseline Evaluation..."
python run.py --eval test \
    --dataset $DATASET \
    --sm sim \
    --num_examples $NUM_EXAMPLES \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --val_size $VAL_SIZE \
    2>&1 | tee results/test/knn_test.log

if [ $? -eq 0 ]; then
    echo "✓ kNN baseline test PASSED"
else
    echo "✗ kNN baseline test FAILED"
    exit 1
fi

# Test 3: PPO training (minimal)
echo "[Test 3/3] Testing PPO Training (minimal)..."
python run.py --train \
    --dataset $DATASET \
    --rl_algo ppo \
    --model_type lstm \
    --model_name results/test/ppo_test \
    --num_examples $NUM_EXAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 1e-4 \
    --hidden_size 256 \
    --generator_model $GENERATOR_MODEL \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    2>&1 | tee results/test/ppo_test.log

if [ $? -eq 0 ]; then
    echo "✓ PPO training test PASSED"
else
    echo "✗ PPO training test FAILED"
    exit 1
fi

echo "=========================================="
echo "All Tests PASSED! ✓"
echo "Your setup is ready for full experiments."
echo ""
echo "Next steps:"
echo "  1. Run: ./experiments/run_main_experiment_fast.sh"
echo "  2. Run: ./experiments/run_ablation_fast.sh"
echo "  3. Run: ./experiments/run_sensitivity_fast.sh"
echo "=========================================="