# Sequential RL RAG

Models RAG retrieval as a sequential decision making process. This project uses an RL agent with dense intermediate marginal rewards and LLM-as-a-judge reward scoring.

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

Currently, we have only done tests with GSM8K. [Download the dataset](https://github.com/openai/grade-school-math) and place the folder in this repo's root folder.

## Run

Due to optimizations to LLM inference (vLLM + caching), training will take around a few hours for the baseline case on a 3090 GPU using Qwen-1.5B. Implementations using intermediate rewards or LLM-as-a-judge will take longer because more LLM generations are required for the reward calculations.

### Training:

Other reward calculations can be selected with the `--reward` argument. The base setup is to train for 2500 examples per epoch for the default 20 epochs. The model will be stored by the value passed into the `--model_name` argument and will be used for evaluation. We use Qwen-2.5-1.5B-Instruct model for all of our experiments; other model sizes (e.g. 0.5B) can be tested by changing `--generator_model`.

```
# sparse rewards
python3 run.py --train --rl_algo ppo_simple --reward exact --dataset gsm8k --model_name gsm8k_ppo_base --generator_model Qwen/Qwen2.5-1.5B-Instruct --e_coef .1 --batch_size 200 --train_size 5000 --corpus_size 200 --soft_prompt_len 20 --val_size 500 --wandb
```

Intermediate rewards can be enabled by adding `--int_reward_margin` to the command.
```
# intermediate marginal rewards
python3 run.py --train --rl_algo ppo_simple --reward exact --int_reward_margin --gamma 0.9 --dataset gsm8k --model_name gsm8k_ppo_int --generator_model Qwen/Qwen2.5-1.5B-Instruct --e_coef .1 --batch_size 200 --train_size 5000 --corpus_size 200 --soft_prompt_len 20 --val_size 500 --wandb
```

LLM-as-a-judge reward scoring can be enabled by changing `--reward` from "exact" to "judge_exact." The judge ratio is controlled by `--cr_coef`.
```
# intermediate marginal rewards with LLM-as-a-judge
python3 run.py --train --rl_algo ppo_simple --reward judge_exact --cr_coef 0.5 --int_reward_margin --gamma 0.9 --dataset gsm8k --model_name gsm8k_ppo_judge --generator_model Qwen/Qwen2.5-1.5B-Instruct --e_coef 0.1 --batch_size 200 --train_size 2500 --epochs 20 --corpus_size 200 --soft_prompt_len 20 --val_size 500 --wandb
```

### Test:

The retriever's sampling method can be changed using the `--sm` argument. The default is `softmax`, but it can also be evaluated using `vf` for a value-function beam-search approach. We found this to generally just perform worse. We can also get a good baseline of the generator model and dataset using the `exhaustive` sampling method (to make this tractable we reduce the corpus size and number of documents retrieved).

Baseline methods generally do not require training, just set the sampling method.
```
# random
python3 run.py --eval test --sm random --dataset gsm8k --generator_model Qwen/Qwen2.5-1.5B-Instruct --batch_size 200

# exhaustive
python3 run.py --eval test --dataset gsm8k --soft_prompt_len 20 --generator_model Qwen/Qwen2.5-1.5B-Instruct --wandb --batch_size 200 --sm exhaustive --val_corpus_size 100 --num_examples 1

# similarity
python3 run.py --eval test --sm sim --dataset gsm8k --generator_model Qwen/Qwen2.5-1.5B-Instruct --batch_size 200
```

For policy-based evaluation, the default sampling method, and the one we used for all of our results, is `softmax`. It can also be evaluated using `vf` for a value-function beam-search approach. We found this to generally just perform worse.

For evaluating policy-based retrievers, make sure to first train it and set `--model_name`. 
```
python3 run.py --eval test --rl_algo ppo_simple --dataset gsm8k --model_name <MODEL_NAME> --generator_model Qwen/Qwen2.5-1.5B-Instruct --batch_size 200 --soft_prompt_len 20 --wandb
```

## AI Usage
We utilized AI coding tools like Cursor for general coding assistance. This was primarily used for intelligent autocomplete and is just the base IDE that I use.

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
