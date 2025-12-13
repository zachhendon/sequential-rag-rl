# Sequential RL RAG

Models RAG retrieval as a sequential decision making process. This project uses an RL agent with dense intermediate marginal rewards and LLM-as-a-judge reward scoring.

This project is built upon [RetICL (Retrieval for In-Context Learning)](https://arxiv.org/abs/2305.14502), a reinforcement learning-based method for the joint retrieval of in-context learning examples. The primary component is a recurrent neural network that jointly represents a problem and a group of examples, along with a bilinear activation that ranks subsequent examples.


## Prerequisites

### System Requirements

- **Python**: 3.9+ (tested on Python 3.9.1 and 3.11)
- **CUDA**: GPU with CUDA support (required for vLLM)
- **GPU Memory**: At least 8GB VRAM recommended (RTX 3090 or better)
- **Operating System**: Linux (tested on Ubuntu)

### Required Software

- CUDA toolkit (for GPU support)
- Git (to clone the repository)
- Conda (optional, for conda environment)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd sequential-rag-rl
```

### Step 2: Create Virtual Environment

**Option A: Using Conda (Recommended)**

```bash
conda create -n sequential-rag-rl python=3.11
conda activate sequential-rag-rl
```

**Option B: Using venv**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Note**: On Windows, use `venv\Scripts\activate` instead.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Important**: This will install PyTorch, vLLM, and other dependencies. Installation may take 10-15 minutes.

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see:
- PyTorch version: 2.8.0 (or similar)
- CUDA available: True (if GPU is properly configured)

---

## Data Setup

### GSM8K Dataset (Default)

The GSM8K dataset is already included in this repository under `grade-school-math/`. No additional download is needed.

**Alternative**: [Download the dataset](https://github.com/openai/grade-school-math) and place the folder in this repo's root folder if needed.

### Other Datasets (Optional)

If you want to use other datasets, download them and place in the repository root:

- **TabMWP**: https://github.com/lupantech/PromptPG/tree/main/data/tabmwp
- **QASC**: http://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz
- **CommonsenseQA**: https://www.tau-nlp.sites.tau.ac.il/commonsenseqa
- **ECQA**: https://github.com/dair-iitd/ECQA-Dataset
- **SVAMP**: https://github.com/arkilpatel/SVAMP


---

## Quick Start

### Option 1: Run Pre-configured Experiments (Recommended)

We provide shell scripts for easy execution:

#### Main Experiment (Compare RL methods vs baselines)

```bash
./run_main_experiment_fast.sh 0.5B

./run_main_experiment_fast.sh 1.5B
```

This will:
1. Run random baseline
2. Run kNN baseline
3. Train 4 RL methods (REINFORCE w/ Baseline, Actor-Critic, PPO, PPO Independent)
4. Save results to `results/main_experiment/`

#### Sensitivity Analysis

```bash
./run_sensitivity.sh 0.5B

# Run specific hyperparameter sensitivity
./run_sensitivity.sh 0.5B learning_rate
./run_sensitivity.sh 0.5B num_examples
```

#### Ablation Studies

```bash
# Run ablation experiments
./run_ablation.sh 0.5B
```

**Note**: Make scripts executable if needed:
```bash
chmod +x run_main_experiment_fast.sh run_sensitivity.sh run_ablation.sh
```

---

## Experiment Types

This repository supports two main types of experiments:

1. **Main Experiments**: Compare different RL algorithms (PPO, Actor-Critic, REINFORCE) and architectures
2. **Reward Ablation Experiments**: Study different reward types (sparse, intermediate marginal, LLM-as-a-judge)

---

## Main Experiments: RL Algorithm Comparison

These experiments compare different RL algorithms and architectures using standard sparse rewards.

### Training

**Basic Training Command**:
```bash
python run.py --train \
    --dataset gsm8k \
    --rl_algo ppo \
    --model_type lstm \
    --model_name results/my_model \
    --num_examples 4 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --hidden_size 512 \
    --generator_model Qwen/Qwen2.5-0.5B \
    --max_gen_tokens 256 \
    --gpu_memory_utilization 0.3 \
    --train_size 1000 \
    --val_size 200
```

**With Weights & Biases Logging**:
```bash
python run.py --train \
    --dataset gsm8k \
    --rl_algo ppo \
    --model_type lstm \
    --model_name results/my_model \
    --num_examples 4 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --hidden_size 512 \
    --generator_model Qwen/Qwen2.5-0.5B \
    --max_gen_tokens 256 \
    --gpu_memory_utilization 0.3 \
    --train_size 1000 \
    --val_size 200 \
    --wandb
```

**Complete Workflow Example**:
```bash

python run.py --train \
    --dataset gsm8k \
    --rl_algo ac \
    --model_type lstm \
    --model_name results/ac_lstm_0.5B \
    --num_examples 4 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --hidden_size 512 \
    --generator_model Qwen/Qwen2.5-0.5B \
    --max_gen_tokens 256 \
    --gpu_memory_utilization 0.3 \
    --train_size 1000 \
    --val_size 200 \
    --wandb

python run.py --eval test \
    --dataset gsm8k \
    --rl_algo ac \
    --model_type lstm \
    --model_name results/ac_lstm_0.5B \
    --num_examples 4 \
    --generator_model Qwen/Qwen2.5-0.5B \
    --max_gen_tokens 256 \
    --gpu_memory_utilization 0.3
```

### Available RL Algorithms

- `ppo`: Proximal Policy Optimization
- `ac`: Actor-Critic
- `rwb`: REINFORCE with Baseline
- `reinforce`: Vanilla REINFORCE

### Available Model Architectures

- `lstm`: LSTM-based sequential model (recommended)
- `rnn`: RNN-based sequential model
- `attn`: Attention-based model
- `ind`: Independent (non-sequential) model

### Model Size Recommendations

**For 0.5B Model (Qwen/Qwen2.5-0.5B)**:
- `--batch_size 32`
- `--gpu_memory_utilization 0.3`
- Faster training (~10-15 min per experiment)

**For 1.5B Model (Qwen/Qwen2.5-1.5B)**:
- `--batch_size 24`
- `--gpu_memory_utilization 0.4`
- Slower training (~20-30 min per experiment)

---

## Reward Ablation Experiments

These experiments study different reward types: sparse rewards, intermediate marginal rewards, and LLM-as-a-judge reward scoring.

**Note**: Due to optimizations to LLM inference (vLLM + caching), training will take around a few hours for the baseline case on a 3090 GPU using Qwen-1.5B-Instruct. Implementations using intermediate rewards or LLM-as-a-judge will take longer because more LLM generations are required for the reward calculations.

### Training with Different Reward Types

#### 1. Sparse Rewards (Base Setup)

The base setup trains for 2500 examples per epoch for the default 20 epochs. The model will be stored by the value passed into the `--model_name` argument and will be used for evaluation.

```bash
python run.py --train \
    --rl_algo ppo_simple \
    --reward exact \
    --dataset gsm8k \
    --model_name gsm8k_ppo_base \
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \
    --e_coef 0.1 \
    --batch_size 200 \
    --train_size 5000 \
    --corpus_size 200 \
    --soft_prompt_len 20 \
    --val_size 500 \
    --wandb
```

#### 2. Intermediate Marginal Rewards

Intermediate rewards can be enabled by adding `--int_reward_margin` to the command.

```bash
python run.py --train \
    --rl_algo ppo_simple \
    --reward exact \
    --int_reward_margin \
    --gamma 0.9 \
    --dataset gsm8k \
    --model_name gsm8k_ppo_int \
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \
    --e_coef 0.1 \
    --batch_size 200 \
    --train_size 5000 \
    --corpus_size 200 \
    --soft_prompt_len 20 \
    --val_size 500 \
    --wandb
```

#### 3. LLM-as-a-Judge Reward Scoring

LLM-as-a-judge reward scoring can be enabled by changing `--reward` from `exact` to `judge_exact`. The judge ratio is controlled by `--cr_coef`.

```bash
python run.py --train \
    --rl_algo ppo_simple \
    --reward judge_exact \
    --cr_coef 0.5 \
    --int_reward_margin \
    --gamma 0.9 \
    --dataset gsm8k \
    --model_name gsm8k_ppo_judge \
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \
    --e_coef 0.1 \
    --batch_size 200 \
    --train_size 2500 \
    --epochs 20 \
    --corpus_size 200 \
    --soft_prompt_len 20 \
    --val_size 500 \
    --wandb
```

**Key Arguments for Reward Experiments**:
- `--reward exact`: Use exact match reward (sparse)
- `--reward judge_exact`: Use LLM-as-a-judge reward scoring
- `--int_reward_margin`: Enable intermediate marginal rewards
- `--cr_coef`: Confidence reward coefficient (for LLM-as-a-judge)
- `--gamma`: Reward discount factor (important for intermediate rewards)
- `--soft_prompt_len`: Length of soft prompts for encoder
- `--corpus_size`: Number of samples in training corpus

**Note**: Other model sizes (e.g., 0.5B) can be tested by changing `--generator_model` to `Qwen/Qwen2.5-0.5B-Instruct`.

---

## Evaluation and Baselines

### Sampling Methods

The retriever's sampling method can be changed using the `--sm` argument:

- `softmax`: Default policy-based sampling (recommended)
- `vf`: Value-function beam-search approach (generally performs worse)
- `exhaustive`: Exhaustive search (requires reduced corpus size)
- `random`: Random baseline
- `sim`: Similarity-based (kNN) baseline

### Baseline Methods

Baseline methods generally do not require training, just set the sampling method.

#### Random Baseline

```bash
python run.py --eval test \
    --sm random \
    --dataset gsm8k \
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \
    --batch_size 200
```

#### Similarity (kNN) Baseline

```bash
python run.py --eval test \
    --sm sim \
    --dataset gsm8k \
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \
    --batch_size 200
```

#### Exhaustive Baseline

To make exhaustive search tractable, we reduce the corpus size and number of documents retrieved:

```bash
python run.py --eval test \
    --dataset gsm8k \
    --soft_prompt_len 20 \
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \
    --wandb \
    --batch_size 200 \
    --sm exhaustive \
    --val_corpus_size 100 \
    --num_examples 1
```

### Policy-Based Evaluation

For evaluating policy-based retrievers, make sure to first train it and set `--model_name`. The default sampling method is `softmax`, which we used for all of our results.

```bash
python run.py --eval test \
    --rl_algo ppo_simple \
    --dataset gsm8k \
    --model_name <MODEL_NAME> \
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \
    --batch_size 200 \
    --soft_prompt_len 20 \
    --wandb
```

**Alternative**: You can also use `--sm vf` for value-function beam-search, though we found this generally performs worse.

---

## Visualization

### Generate Results Visualizations

After running experiments, generate publication-quality figures:

```bash
python visualize_1.5B_results.py
```

This creates 5 figures in `results/figures/`:
- `figure1_method_comparison.pdf/png`: All methods comparison
- `figure2_train_val.pdf/png`: Generalization analysis
- `figure3_architecture.pdf/png`: Architecture comparison
- `figure4_improvement.pdf/png`: Improvement over baseline
- `figure5_summary.pdf/png`: Comprehensive summary (recommended for papers)

### Analyze Sensitivity Results

```bash
python analyze_sensitivity.py 0.5B

python analyze_sensitivity.py 1.5B
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `--gpu_memory_utilization` (try 0.2 or 0.25)
- Reduce `--batch_size` (try 16 or 8)
- Use smaller model: `Qwen/Qwen2.5-0.5B` instead of `Qwen/Qwen2.5-1.5B`

#### 2. Module Not Found

**Error**: `ModuleNotFoundError: No module named 'vllm'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### 3. Dataset Not Found

**Error**: `Dataset gsm8k not found`

**Solution**: Ensure GSM8K data is in `grade-school-math/` directory (already included in repo)

#### 4. Permission Denied for Scripts

**Error**: `Permission denied: ./run_main_experiment_fast.sh`

**Solution**:
```bash
chmod +x run_main_experiment_fast.sh run_sensitivity.sh run_ablation.sh
```

#### 5. vLLM Initialization Failed

**Error**: `Engine core initialization failed`

**Solutions**:
- Check GPU availability: `nvidia-smi`
- Reduce `--gpu_memory_utilization`
- Restart Python process
- Check CUDA version compatibility

### Getting Help

1. Check logs in `results/` directory for detailed error messages
2. Verify GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check CUDA version: `nvcc --version`
4. Review command-line arguments: `python run.py --help`

---

## Advanced Usage

### Custom Training Configuration

You can customize many hyperparameters. See all options:

```bash
python run.py --help
```

### Key Hyperparameters

- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Training batch size (default: 32, or 200 for reward experiments)
- `--hidden_size`: LSTM hidden dimension (default: 512)
- `--e_coef`: Entropy coefficient for exploration (default: 0.01)
- `--gamma`: Reward discount factor (default: 0.99, important for intermediate rewards)
- `--num_examples`: Number of examples to retrieve (default: 4)
- `--cr_coef`: Confidence reward coefficient for LLM-as-a-judge (default: varies)
- `--soft_prompt_len`: Length of soft prompts for encoder (default: 20)
- `--corpus_size`: Number of samples in training corpus (default: 200)

### Available Datasets

- `gsm8k`: Grade School Math 8K (default, included)
- `tabmwp`: TabMWP
- `qasc`: QASC
- `commonsense_qa`: CommonsenseQA
- `ecqa`: ECQA
- `svamp`: SVAMP
- `math`: MATH
- `mtop`: MTOP
- `ag_news`: AG News

### Saving and Loading Models

Models are automatically saved to the path specified by `--model_name`:
- Training checkpoints: `{model_name}.pt`
- Logs: `{model_name}.log` (if using shell scripts)

To load a trained model for evaluation:
```bash
python run.py --eval test \
    --model_name results/my_model \
    --rl_algo ppo \
    --model_type lstm \
    ...
```


## AI Usage

We utilized AI coding tools like Cursor for general coding assistance. This was primarily used for intelligent autocomplete and is just the base IDE that we use.

We also used AI to check for any grammatical or logical errors in our report.


## Acknowledgements

This project is built upon the work proposed in **RetICL** (Scarlatos & Lan, 2024). We appreciate the authors for open-sourcing their code.

```
@misc{scarlatos2024reticl,
      title={RetICL: Sequential Retrieval of In-Context Examples with Reinforcement Learning},
      author={Alexander Scarlatos and Andrew Lan},
      year={2024},
      eprint={2305.14502},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

