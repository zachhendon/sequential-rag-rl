# Sequential RL RAG

Models RAG retrieval as a sequential decision making process. This project uses an RL agent with dense intermediate marginal rewards.

## Setup

### Python Environment

Create and activate a Conda environment with python 3.11:

```
conda create -n sequential-rag-rl python=3.11
conda activate sequential-rag-rl
```

Install dependencies with pip:

```
pip install -r requirements.txt
```

### Data

Currenty, we have only done tests with GSM8K. [Download the dataset](https://github.com/openai/grade-school-math) and place the folder in this repo's root folder.

## Run

Due to optimizations to LLM inference (vLLM + caching), training will take around a few hours for the baseline case on a 3090 GPU using Qwen-0.5B. Implementations using intermediate rewards or LLM-as-a-judge will take longer because more LLM generations are required for the reward calculations.

### Training:

The reward method is our primary modification. Intermediate rewards can be enabled by adding `--int_reward_margin` to the command and other reward calculations can be selected with the `--reward` argument. The base setup is to train for 5000 examples per epoch for the default 50 epochs. The model will be stored by the value passed into the `--model_name` argument. We use Qwen-2.5 models for all of our experiments; other model sizes (e.g. 1.5B, 3B, 7B) can be tested by changing `--generator_model`.

```
# baseline - no intermediate rewards
python3 run.py --train --rl_algo ppo_simple --reward exact --dataset gsm8k --model_name gsm8k_ppo_base --generator_model Qwen/Qwen2.5-0.5B-Instruct --e_coef .1 --batch_size 128 --train_size 5000 --corpus_size 200 --soft_prompt_len 20 --val_size 500 --wandb

# intermediate marginal rewards
python3 run.py --train --rl_algo ppo_simple --reward exact --int_reward_margin --gamma 0.9 --dataset gsm8k --model_name gsm8k_ppo_int -generator_model Qwen/Qwen2.5-0.5B-Instruct --e_coef .1 --batch_size 128 --train_size 5000 --corpus_size 200 --soft_prompt_len 20 --val_size 500 --wandb
```

### Test:

The retriever's sampling method can be changed using the `--sm` argument. The default is `softmax`, but it can also be evaluated using `vf` for a value-function beam-search approach. We can also get a good baseline of the generator model and dataset using the `exhaustive` sampling method (to make this tractable we reduce the corpus size and number of documents retrieved).

```
# policy evaluation
python3 run.py --eval test --rl_algo ppo_simple --dataset gsm8k --model_name <MODEL_NAME> --generator_model Qwen/Qwen2.5-0.5B-Instruct --batch_size 128 --soft_prompt_len 20 --wandb

# baseline exhaustive evaluation
python3 run.py --eval test --dataset gsm8k --generator_model Qwen/Qwen2.5-0.5B-Instruct --batch_size 128 --soft_prompt_len 20 --sm exhaustive --val_corpus_size 100 --num_examples 1 --wandb
```

## Ackowledgements

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
