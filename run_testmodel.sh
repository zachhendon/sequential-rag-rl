#!/bin/bash
# Alternative Small Models for Fast Experimentation
# Use this script to test different small models

# Option 1: Qwen2.5-0.5B (Recommended - very fast)
# ~500M parameters, good instruction following
MODEL_QWEN_05B="Qwen/Qwen2.5-0.5B"
MODEL_QWEN_05B_INSTRUCT="Qwen/Qwen2.5-0.5B-Instruct"

# Option 2: Qwen2.5-1.5B (Slightly larger, better quality)
MODEL_QWEN_15B="Qwen/Qwen2.5-1.5B"
MODEL_QWEN_15B_INSTRUCT="Qwen/Qwen2.5-1.5B-Instruct"

# Option 3: TinyLlama (1.1B, good for math)
MODEL_TINYLLAMA="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Option 4: Phi-2 (2.7B, Microsoft's small model)
MODEL_PHI2="microsoft/phi-2"

# Option 5: SmolLM (135M - extremely fast but lower quality)
MODEL_SMOLLM="HuggingFaceTB/SmolLM-135M"
MODEL_SMOLLM_360M="HuggingFaceTB/SmolLM-360M"

# Option 6: GPT-2 variants (for debugging)
MODEL_GPT2="gpt2"              # 124M
MODEL_GPT2_MEDIUM="gpt2-medium" # 355M

echo "=========================================="
echo "Testing Available Small Models"
echo "=========================================="

# Test function
test_model() {
    MODEL=$1
    echo "Testing: ${MODEL}..."
    python run.py --eval test \
        --dataset gsm8k \
        --sm random \
        --num_examples 2 \
        --generator_model $MODEL \
        --max_gen_tokens 64 \
        --val_size 5 \
        2>&1 | head -20
    
    if [ $? -eq 0 ]; then
        echo "✓ ${MODEL} works!"
        return 0
    else
        echo "✗ ${MODEL} failed"
        return 1
    fi
}

# Uncomment the model you want to test:
test_model $MODEL_QWEN_05B
# test_model $MODEL_QWEN_05B_INSTRUCT
# test_model $MODEL_QWEN_15B
# test_model $MODEL_TINYLLAMA
# test_model $MODEL_PHI2
# test_model $MODEL_SMOLLM_360M
# test_model $MODEL_GPT2

echo "=========================================="
echo "Model Recommendations:"
echo ""
echo "For FAST experiments (debugging):"
echo "  - Qwen/Qwen2.5-0.5B (recommended)"
echo "  - HuggingFaceTB/SmolLM-360M"
echo ""
echo "For BETTER quality (final results):"
echo "  - Qwen/Qwen2.5-1.5B-Instruct"
echo "  - TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo ""
echo "For PRODUCTION (if you have GPU memory):"
echo "  - mistralai/Mistral-7B-v0.1 (default)"
echo "  - meta-llama/Llama-2-7b-hf"
echo "=========================================="