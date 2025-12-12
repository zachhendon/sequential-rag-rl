from typing import List
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
import re

from reticl.models.retriever import retriever_model, Retriever
from reticl.models.generator import VLLMGenerator
from reticl.data_loading.data_types import DatasetConfig
from reticl.data_loading.reticl_dataset import RetICLDataset, Collator, CollatedBatch, filter_batch
from reticl.training.replay_buffer import ReplayBuffer
from reticl.constants import SamplingMethod, RLAlgorithm, Reward, LRSchedule
from reticl.utils import TrainOptions, device, is_pg

class TrainSeqRAG:
    def __init__(self, generator: VLLMGenerator, dataset_config: DatasetConfig):
        self.generator = generator
        self.dataset_config = dataset_config
    
    def _construct_judge_prompts(self, batch: CollatedBatch, predictions: List[dict]):
        prompts = []
        for meta, pred in zip(batch["meta_data"], predictions):
            # TODO: assumes GSM8K dataset for now
            question = meta["question"]
            answer = meta["answer"]
            model_response = pred["text"]
            prompt = (
                f"### Role\n"
                f"You are an expert evaluator for a question answering system\n\n"
                f"### Instructions\n"
                f"Your task is to evaluate the model response based on the user question, the correct answer, and the model response and give a score between 0 and 100. "
                f"100 is the highest score and reflects a perfect answer with perfect reasoning. "
                f"0 is the lowest score and reflects a completely incorrect answer with no progress towards the correct answer. "
                f"A score of 50 is a mediocre score and reflects an answer that is partially correct with some key mistakes. "
                f"The model should aim to get the correct final answer and logiclly explain every step accurately.\n\n"
                f"### Output format\n"
                f"Your response should consist of only these two sections: reasoning and score. The response should end once you have provided the final score. "
                f"Your reasoning should be focused and logical. The score should be solely based on the accuracy and reasoning of the response and no other factors.\n"
                f"1. Reasoning: Explain your thinking process and explain what the model did well and what it did poorly. "
                f"During the reasoning process, you should generate a score and then explain your reasoning for that score. "
                f"Explain why you gave the response that specific score and why it is not higher or lower. Try to keep the reasoning concise and to the point.\n"
                f"2. Score: Give a final score between 0 and 100 based on the reasoning. "
                f"The output format of the score should be a single integer between 0 and 100 and it should be wrapped in <score> tags. "
                f"Examples: <score>90</score>, <score>0</score>, <score>40</score>\n"
                f"3. Output verification: Verify that you have provided a score for the response and that it is wrapped in <score> tags.\n\n"
                f"### Input data\n"
                f"User question: {question}\n"
                f"Correct answer: {answer}\n"
                f"Model response: {model_response}\n\n"
                f"### Evaluation\n"
            )
            prompts.append(prompt)
        return prompts
    
    def _extract_judge_score(self, response: str):
        score_match = re.findall(r"<score>(\d+)</score>", response["text"])
        if len(score_match) > 0:
            score = int(score_match[-1]) # extract the last score if there are multiple
        else:
            score = 50
        if score > 100 or score < 0:
            score = 50
        return (score - 50) / 50

    def get_predictions(self, batch: CollatedBatch, step: int = None, generator=None, dataset_config=None):
        # Generate predictions given retrieved context and check correctness
        generator = generator or self.generator
        dataset_config = dataset_config or self.dataset_config
        
        if batch["outputs"] is not None:
            predictions = batch["outputs"]
        else:
            if step is None:
                prompts = batch["first_prompts"]
            else:
                prompts = [sub_prompts[step] for sub_prompts in batch["sub_prompts"]]
            predictions = generator.generate(prompts)
        if dataset_config.get("check_correct_batch"):
            correct = dataset_config["check_correct_batch"](
                batch["meta_data"], [pred["text"] for pred in predictions])
        else:
            correct = torch.Tensor([
                dataset_config["check_correct"](src_meta_data, pred["text"])
                for src_meta_data, pred in zip(batch["meta_data"], predictions)
            ])
        return predictions, correct

    def get_rewards(self, batch: CollatedBatch, options: TrainOptions, device: torch.device, anneal: float = 1.0, step: int = None, generator=None, dataset_config=None):
        rewards = None

        if options.reward in (Reward.EXACT.value, Reward.CONF.value, Reward.EXACT_AND_BLEU.value, Reward.JUDGE.value, Reward.JUDGE_EXACT.value):
            predictions, correct = self.get_predictions(batch, step, generator, dataset_config)
            correct = correct.to(device)

            if options.reward == Reward.EXACT.value:
                # Reward is 1 if prediction is correct, -1 otherwise
                rewards = 2 * correct - 1

            elif options.reward == Reward.CONF.value:
                ppl = torch.tensor([-pred["nll"] for pred in predictions]).exp()
                cr_coef = options.cr_coef * anneal
                rewards = 2 * (correct * (1 - cr_coef) + ppl * cr_coef) - 1
            elif options.reward in (Reward.JUDGE.value, Reward.JUDGE_EXACT.value):
                judge_prompts = self._construct_judge_prompts(batch, predictions)
                judge_responses = self.generator.generate(judge_prompts)
                judge_scores = [self._extract_judge_score(response) for response in judge_responses]
                rewards = torch.tensor(judge_scores).to(device)

                if options.reward == Reward.JUDGE_EXACT.value:
                    # Take a weighted average ofthe judge score and the correct reward
                    cr_coef = options.cr_coef * anneal
                    correct_rewards = 2 * correct - 1
                    rewards = correct_rewards * (1 - cr_coef) + rewards * cr_coef
            rewards = rewards.to(device)
        return rewards, correct

    def get_returns(self, batch: CollatedBatch, options: TrainOptions, anneal: float = 1.0, train: bool = False, generator=None, dataset_config=None):
        # Calculate rewards and returns for batch - rewards given at eos actions
        batch_size, max_seq_len = batch["example_encodings"].shape[:2]

        if train and options.int_reward_margin:
            # Compute returns as discounted marginal rewards
            rewards = torch.zeros((batch_size, max_seq_len), device=device)
            returns = torch.zeros((batch_size, max_seq_len), device=device)

            cur_rewards, _ = self.get_rewards(batch, options, device, step=None, generator=self.generator, dataset_config=self.dataset_config)
            for step in range(max_seq_len - 1):
                new_rewards, correct = self.get_rewards(batch, options, device, step=step, generator=self.generator, dataset_config=self.dataset_config)
                rewards[:, step] = new_rewards
                marginal_rewards = new_rewards - cur_rewards
                returns[:, step] = marginal_rewards * options.gamma ** step
                cur_rewards = new_rewards
        else:
            # Just compute the final reward
            final_rewards, correct = self.get_rewards(batch, options, device, anneal, max_seq_len - 2, generator, dataset_config)
            rewards = torch.zeros((batch_size, max_seq_len), device=device)
            rewards[torch.arange(batch_size), batch["seq_len"] - 1] = final_rewards
            returns = rewards.clone()
        returns = returns.view(-1)
        return returns, rewards, correct

    def get_td_error(self, value_estimates: torch.Tensor, rewards: torch.Tensor, options: TrainOptions):
        batch_size = rewards.shape[0]
        # Append 0 to value estimates for terminal state
        v_t = F.pad(value_estimates.detach().view(batch_size, -1), (0, 1))
        # TD error: r_t + gamma * v_(t+1) - v_t
        return (rewards + options.gamma * v_t[:, 1:] - v_t[:, :-1]).view(-1)

    def get_gae(self, value_estimates: torch.Tensor, rewards: torch.Tensor, options: TrainOptions):
        batch_size = rewards.shape[0]
        # GAE: sum_{i=t}^{T} (r_t + gamma * v_(i+1) - v_i) * (gamma * lam)^(T-t)
        gae = self.get_td_error(value_estimates, rewards, options).view(batch_size, -1)
        for t in range(gae.shape[1] - 2, -1, -1):
            gae[:, t] += options.gamma * options.lam * gae[:, t + 1]
        return gae.view(-1)

    def get_entropy(self, activations: torch.Tensor):
        # H_t = -sum(pi(s,.) * log(pi(s,.)))
        # Take average over batch and all time steps
        action_distro = torch.softmax(activations, dim=-1).clip(1e-35)
        entropy = -torch.sum(action_distro * torch.log(action_distro), dim=-1)
        # Normalize by maximum entropy so coefficient is independent of action space size
        return entropy / torch.log(torch.tensor(action_distro.shape[-1]))

    @staticmethod
    def get_optim(models: List[Retriever], options: TrainOptions, checkpoint = None):
        all_named_params = []
        for model in models:
            all_named_params += list(model.named_parameters())
        retriever_params = [param for name, param in all_named_params if "encoder" not in name]
        encoder_params = [param for name, param in all_named_params if "encoder" in name]
        optimizer = torch.optim.AdamW([
            {"params": retriever_params},
            {"params": encoder_params, "lr": options.encoder_lr or options.lr}
        ], lr=options.lr, weight_decay=options.wd, eps=options.adam_eps)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint)
        if options.lr_sched == LRSchedule.LINEAR.value:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer, start_factor=1.0, end_factor=0.0, total_iters=options.epochs)
        elif options.lr_sched == LRSchedule.CYCLE.value:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=[group["lr"] for group in optimizer.param_groups], total_steps=options.epochs, pct_start=0.1, anneal_strategy="linear"
            )
        else:
            lr_scheduler = None
        return optimizer, lr_scheduler, retriever_params + encoder_params

    def polyak_update(self, source: torch.nn.Module, target: torch.nn.Module, tau: float):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def validation_loop(self, retriever: Retriever, best_model: Retriever, best_val_accuracy: float,
                        val_set: RetICLDataset, val_loader: DataLoader, train_stats: dict,
                        run, options: TrainOptions):
        with torch.no_grad():
            # Re-compute corpus encodings for validation set if training encoder
            if retriever.encoder is not None:
                val_set.compute_corpus_encodings()

            # Get average reward on validation set
            val_reward = 0
            val_correct = 0
            val_entropy = 0
            val_num_examples = 0
            val_example_set = set()
            retriever.eval()
            pbar_len = len(val_set)
            if options.reward in (Reward.JUDGE.value, Reward.JUDGE_EXACT.value):
                pbar_len = len(val_set) * 2
            self.generator.reset_pbar(pbar_len, "Validation")
            for batch in val_loader:
                for example_idx in batch["policy_example_indices"].view(-1):
                    val_example_set.add(example_idx.item())

                _, rewards, correct = self.get_returns(batch, options, train_stats["supp_reward_anneal"])
                val_reward += rewards.detach().cpu().numpy().sum()
                val_correct += correct.detach().cpu().numpy().sum()
                val_num_examples += (batch["seq_len"] - 1).sum().item()

                if is_pg(options):
                    batch_size, max_seq_len = batch["example_encodings"].shape[:2]
                    ent_mask = torch.arange(max_seq_len).expand(batch_size, -1) >= batch["seq_len"].unsqueeze(1)
                    if max_seq_len > options.num_examples:
                        ent_mask[:, options.num_examples] = True
                    activations, _ = retriever(**batch)
                    val_entropy += self.get_entropy(activations)[~ent_mask.view(-1)].mean().item()

        # Report stats on current epoch
        avg_loss = np.mean(train_stats["losses"])
        avg_reward = train_stats["total_reward"] / train_stats["train_size"]
        avg_train_accuracy = train_stats["total_correct"] / train_stats["train_size"]
        avg_val_reward = val_reward / len(val_set)
        avg_val_accuracy = val_correct / len(val_set)
        avg_num_examples = train_stats["train_num_examples"] / train_stats["train_size"]
        avg_val_num_examples = val_num_examples / len(val_set)
        avg_val_entropy = val_entropy / len(val_loader) if val_entropy else None
        if run:
            run.log({
                "loss": avg_loss,
                "vf_loss": np.mean(train_stats["vf_losses"]) if train_stats["vf_losses"] else None,
                "reward": avg_reward,
                "train_accuracy": avg_train_accuracy,
                "val_reward": avg_val_reward,
                "val_accuracy": avg_val_accuracy,
                "train_examples_total": len(train_stats["train_example_set"]),
                "train_examples_per": avg_num_examples,
                "val_examples_total": len(val_example_set),
                "val_examples_per": avg_val_num_examples,
                "val_entropy": avg_val_entropy,
                "lr": train_stats["cur_lr"],
                "clip_fraction": np.mean(train_stats["clip_fractions"]) if train_stats["clip_fractions"] else None,
            })
        print(f"Epoch {train_stats['epoch'] + 1}, Loss: {avg_loss:.4f}, "
                f"Reward: {avg_reward:.4f}, Val Reward: {avg_val_reward:.4f}, "
                f"Train Acc: {avg_train_accuracy:.4f}, Val Acc: {avg_val_accuracy:.4f}, "
                f"Train Examples: {len(train_stats['train_example_set'])}, Val Examples: {len(val_example_set)}")

        # Save checkpoint
        # Commented since not properly resuming
        # torch.save({
        #     "retriever": retriever.state_dict(),
        #     "val_est": val_est_model.state_dict() if val_est_model is not None else None,
        #     "optimizer": optimizer.state_dict(),
        #     "rng_state": torch.random.get_rng_state(),
        #     "epoch": epoch + 1,
        # }, checkpoint_path)

        # Save model with best reward on validation set
        if best_val_accuracy is None or avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            print("Best! Saving model...")
            best_model.load_state_dict(retriever.state_dict())
            torch.save(best_model.state_dict(), f"{options.model_name}.pt")
        return best_val_accuracy

    def reset_train_stats(self, epoch: int, cur_lr: float, supp_reward_anneal: float):
        return {
            "epoch": epoch,
            "cur_lr": cur_lr,
            "supp_reward_anneal": supp_reward_anneal,
            "train_size": 0,
            "total_reward": 0,
            "total_correct": 0,
            "train_num_examples": 0,
            "train_example_set": set(),
            "losses": [],
            "vf_losses": [],
            "clip_fractions": [],
        }

    def train(self, train_split: str, dev_split: str, options_dict: dict):
        options = TrainOptions(options_dict)
        assert(options.model_name)
        if options.wandb:
            run = wandb.init(project="reticl", config=options.as_dict())
        else:
            run = None

        # Load checkpoint
        checkpoint_path = f"{options.model_name}_ckpt.pt"
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = None

        # Create/load model(s) and optimizer(s)
        if options.rl_algo == RLAlgorithm.DSAC.value and not options.sep_val_model:
            retriever = retriever_model(options, num_critics=2)
            best_model = retriever_model(options, num_critics=2)
        else:
            retriever = retriever_model(options)
            best_model = retriever_model(options)
        if options.pt_model_name:
            retriever.load_state_dict(torch.load(f"{options.pt_model_name}.pt", map_location=device))
        if checkpoint:
            retriever.load_state_dict(checkpoint["retriever"])
        retriever.train()
        if options.sep_val_model and options.rl_algo != RLAlgorithm.DSAC.value:
            val_est_model = retriever_model(options)
            if checkpoint:
                val_est_model.load_state_dict(checkpoint["val_est"])
            val_est_model.train()
        else:
            val_est_model = None
        optimizer, scheduler, retriever_params = self.get_optim(
            [retriever, val_est_model] if val_est_model is not None else [retriever],
            options, checkpoint["optimizer"] if checkpoint else None
        )

        # Initialize critics/targets/alpha/optimizers for DSAC
        if options.rl_algo == RLAlgorithm.DSAC.value:
            num_critics = 2
            if options.sep_val_model:
                critics = [retriever_model(options, True, False) for _ in range(num_critics)]
                critic_targets = [retriever_model(options, True, False) for _ in range(num_critics)]
                cosps = [self.get_optim([critic], options) for critic in critics]
                critic_optims = [cosp[0] for cosp in cosps]
                critic_schedulers = [cosp[1] for cosp in cosps]
                for critic, target in zip(critics, critic_targets):
                    target.load_state_dict(critic.state_dict())
                    target.eval()
            else:
                cosps = [self.get_optim([retriever], options) for _ in range(num_critics)]
                critic_optims = [cosp[0] for cosp in cosps]
                critic_schedulers = [cosp[1] for cosp in cosps]
                target = retriever_model(options, num_critics=2)
                target.load_state_dict(retriever.state_dict())
                target.eval()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = torch.optim.AdamW([log_alpha], lr=options.lr, weight_decay=options.wd)
            target_entropy = options.e_coef * torch.log(torch.tensor(options.corpus_size))

        # Create train/val datasets/loaders
        dataset = RetICLDataset(self.dataset_config, train_split, retriever, options, True)
        val_set = RetICLDataset(self.dataset_config, dev_split, retriever, options, False)
        val_set.set_greedy(True) # Use greedy retrieval for validation
        data_loader = DataLoader(
            dataset,
            # Actual collate done outside loader so it's easier to collect samples for adding to replay buffer
            collate_fn=lambda batch: batch,
            batch_size=options.batch_size,
            shuffle=True,
            drop_last=False
        )
        collator = Collator(len(dataset.corpus))
        val_loader = DataLoader(
            val_set,
            collate_fn=Collator(len(val_set.corpus)),
            batch_size=options.batch_size,
            shuffle=False,
            drop_last=False
        )

        print("Training...")
        # torch.autograd.set_detect_anomaly(True)
        if options.rl_algo in (RLAlgorithm.PPO.value, RLAlgorithm.PPO_SIMPLE.value):
            previous_model = retriever_model(options)
            previous_model.load_state_dict(retriever.state_dict())
        if options.rl_algo == RLAlgorithm.DSAC.value:
            replay_buffer = ReplayBuffer(options)
        best_val_accuracy = None
        e_coef = 0.0
        if checkpoint:
            torch.random.set_rng_state(checkpoint["rng_state"].cpu())
        starting_epoch = 0 if checkpoint is None else checkpoint["epoch"]
        for epoch in range(starting_epoch, options.epochs):
            # Update hyperparameters
            cur_lr = optimizer.param_groups[0]["lr"]
            if options.sm == SamplingMethod.EPSILON_GREEDY.value:
                dataset.update_epsilon(options.eg_eps * options.expl_decay_rate ** epoch)
            else:
                e_coef = options.e_coef * max(1 - (1 - options.expl_decay_rate) * epoch / options.epochs, 0)
            # TODO: do annealing every step instead of every epoch
            if options.anneal_reward:
                supp_reward_anneal = 1 - epoch / options.epochs
            else:
                supp_reward_anneal = 1.0

            # Sample batch from dataset - example retrieval is also done here (__getitem__ in RetICLDataset)
            train_stats = self.reset_train_stats(epoch, cur_lr, supp_reward_anneal)
            if options.int_reward_margin:
                pbar_len = len(dataset) * (options.num_examples + 1)
            else:
                pbar_len = len(dataset)
            if options.reward in (Reward.JUDGE.value, Reward.JUDGE_EXACT.value):
                pbar_len *= 2
            self.generator.reset_pbar(pbar_len, f"Epoch {epoch+1}/{options.epochs}")
            for batch_idx, raw_batch in enumerate(data_loader):
                retriever.train()
                batch = collator(raw_batch)
                batch_size, max_seq_len = batch["example_encodings"].shape[:2]
                train_stats["train_size"] += batch_size

                # Keep track of used examples
                for example_idx in batch["policy_example_indices"].view(-1):
                    train_stats["train_example_set"].add(example_idx.item())
                train_stats["train_num_examples"] += (batch["seq_len"] - 1).sum().item()

                # Get rewards and returns
                returns, rewards, correct = self.get_returns(batch, options, supp_reward_anneal, True)
                train_stats["total_reward"] += rewards.sum().item()
                if correct is not None:
                    train_stats["total_correct"] += correct.sum().item()

                # Off-policy: add to buffer
                if options.rl_algo == RLAlgorithm.DSAC.value:
                    # Add episodes to buffer
                    replay_buffer.add(raw_batch, rewards)

                    # Don't continue to training if buffer isn't full enough yet
                    if len(replay_buffer) < options.episodes_before_train:
                        continue

                # For methods with no inner loop - get current example activations and value function estimates
                if options.rl_algo not in (RLAlgorithm.DSAC.value, RLAlgorithm.PPO.value):
                    if options.sep_val_model:
                        activations, _ = retriever(**batch)
                        _, value_estimates = val_est_model(**batch)
                    else:
                        activations, value_estimates = retriever(**batch)

                # Loss mask - don't compute loss on padding regions
                loss_mask = torch.arange(max_seq_len).expand(batch_size, -1) >= batch["seq_len"].unsqueeze(1)
                ent_mask = loss_mask.clone()
                if max_seq_len > options.num_examples:
                    # Don't compute entropy when eos is forced
                    ent_mask[:, options.num_examples] = True

                # Calculate loss and backprop
                vf_loss = None
                if options.rl_algo == RLAlgorithm.MCC.value:
                    loss = torch.nn.MSELoss(reduction="none")(activations, returns)

                elif options.rl_algo == RLAlgorithm.REINFORCE.value:
                    # REINFORCE: param = param + lr * G * grad(log(pi[a]))
                    # GD: param = param - lr * grad(loss)
                    # loss = -G * log(pi[a])
                    # pi[a] = softmax(activations)[a]
                    # CEL = -log(softmax(activations)[a])
                    # loss = G * CEL
                    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fn(activations, batch["policy_example_indices"].view(-1))
                    loss = loss * returns.view(-1)

                elif options.rl_algo == RLAlgorithm.RWB.value:
                    pg_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                    pg_loss = pg_loss_fn(activations, batch["policy_example_indices"].view(-1))
                    pg_loss = pg_loss * (returns.view(-1) - value_estimates.detach()) # Don't differentiate w.r.t. baseline
                    vf_loss = torch.nn.MSELoss(reduction="none")(value_estimates, returns)
                    loss = pg_loss + options.v_coef * vf_loss

                elif options.rl_algo == RLAlgorithm.AC.value:
                    td_error = self.get_td_error(value_estimates, rewards, options)
                    pg_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                    pg_loss = pg_loss_fn(activations, batch["policy_example_indices"].view(-1))
                    pg_loss = pg_loss * td_error
                    # (r_t + v_(t+1) - v_t)^2 = ((r_t + v_(t+1) - v_t + v_t) - v_t)^2 = ((td_err + v_t) - v_t)^2
                    vf_loss = torch.nn.MSELoss(reduction="none")(value_estimates, td_error + value_estimates.detach())
                    loss = pg_loss + options.v_coef * vf_loss

                elif options.rl_algo == RLAlgorithm.PPO_SIMPLE.value:
                    # Get policy ratio
                    with torch.no_grad():
                        pi_old_activations, _ = previous_model(**batch)
                    cur_policy_probs = torch.softmax(activations, dim=-1)[
                        torch.arange(batch_size * max_seq_len), batch["policy_example_indices"].view(-1)]
                    old_policy_probs = torch.softmax(pi_old_activations, dim=-1)[
                        torch.arange(batch_size * max_seq_len), batch["policy_example_indices"].view(-1)]
                    ratio = cur_policy_probs / old_policy_probs
                    previous_model.load_state_dict(retriever.state_dict()) # Copy model for next iteration

                    # Get batch-normalized estimated advantage
                    advantage = self.get_gae(value_estimates, rewards, options)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    # Get clip loss
                    clip_loss = -torch.min(ratio * advantage, torch.clip(ratio, 1 - options.ppo_eps, 1 + options.ppo_eps) * advantage)

                    # Get value function loss
                    vf_loss = torch.nn.MSELoss(reduction="none")(value_estimates, returns)

                    # Get final loss
                    if options.sep_val_model:
                        loss = clip_loss
                        vf_loss[loss_mask.view(-1)] = 0
                        vf_loss = vf_loss.sum() / (~loss_mask).sum()
                        vf_loss.backward()
                    else:
                        loss = clip_loss + options.v_coef * vf_loss

                elif options.rl_algo == RLAlgorithm.PPO.value:
                    # Get previous action probabilities
                    with torch.no_grad():
                        pi_old_activations, _ = previous_model(**batch)
                        old_policy_probs = torch.softmax(pi_old_activations, dim=-1)[
                            torch.arange(batch_size * max_seq_len), batch["policy_example_indices"].view(-1)]

                    # Copy model for next iteration
                    previous_model.load_state_dict(retriever.state_dict())

                    # Get estimated advantage
                    with torch.no_grad():
                        if options.sep_val_model:
                            _, value_estimates = val_est_model(**batch)
                        else:
                            _, value_estimates = retriever(**batch)
                    advantage = self.get_gae(value_estimates, rewards, options)
                    # TODO: see if this helps or not
                    # returns = advantage + value_estimates # Use TD(lambda) return as value function target

                    # TODO: mess with hyperparams - batch size controls amount of exploration?
                    for _ in range(options.inner_epochs):
                        idxs = list(range(batch_size * max_seq_len))
                        random.shuffle(idxs)
                        # TODO: try making sub-batches full sequences rather than individual indices
                        # TODO: look into kl early stopping
                        for start_idx in range(0, len(idxs), options.sub_batch_size):
                            sub_batch_idxs = idxs[start_idx : start_idx + options.sub_batch_size]
                            if options.sep_val_model:
                                activations, _ = retriever(**batch)
                                _, value_estimates = val_est_model(**batch)
                            else:
                                activations, value_estimates = retriever(**batch)

                            # Get clip loss
                            sub_batch_loss_mask = loss_mask.view(-1)[sub_batch_idxs]
                            sub_batch_activations = activations[sub_batch_idxs]
                            sub_batch_policy_example_idxs = batch["policy_example_indices"].view(-1)[sub_batch_idxs]
                            sub_batch_cur_probs = torch.softmax(sub_batch_activations, dim=-1)[
                                torch.arange(len(sub_batch_idxs)), sub_batch_policy_example_idxs]
                            sub_batch_old_probs = old_policy_probs[sub_batch_idxs]
                            ratio = sub_batch_cur_probs / sub_batch_old_probs
                            sub_batch_advantage = advantage[sub_batch_idxs]
                            clip_loss = -torch.min(
                                ratio * sub_batch_advantage,
                                torch.clip(ratio, 1 - options.ppo_eps, 1 + options.ppo_eps) * sub_batch_advantage
                            )
                            clip_loss = clip_loss[~sub_batch_loss_mask].mean()
                            train_stats["clip_fractions"].append((torch.abs(ratio - 1) > options.ppo_eps).float().mean().item())

                            # Get value function loss
                            sub_batch_value_estimates = value_estimates[sub_batch_idxs]
                            sub_batch_returns = returns[sub_batch_idxs]
                            vf_loss = torch.nn.MSELoss(reduction="none")(sub_batch_value_estimates, sub_batch_returns)
                            vf_loss = vf_loss[~sub_batch_loss_mask].mean()
                            train_stats["vf_losses"].append(vf_loss.item())

                            # Get final loss
                            if options.sep_val_model:
                                loss = clip_loss
                                vf_loss.backward()
                            else:
                                loss = clip_loss + options.v_coef * vf_loss
                            # Maximize entropy - encourages exploration by flattening action distribution
                            sub_batch_ent_mask = ent_mask.view(-1)[sub_batch_idxs]
                            entropy = self.get_entropy(sub_batch_activations)
                            entropy = entropy[~sub_batch_ent_mask].mean()
                            loss = loss - e_coef * entropy
                            train_stats["losses"].append(loss.item())
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(retriever_params, options.grad_clip)
                            optimizer.step()
                            optimizer.zero_grad()

                elif options.rl_algo == RLAlgorithm.DSAC.value:
                    # Implementation roughly based on https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
                    for _ in range(options.updates_per_batch):
                        batch_size = options.train_batch_size
                        raw_batch, rewards = replay_buffer.sample(batch_size)
                        batch = collator(raw_batch)

                        # Get policy at each state
                        with torch.no_grad():
                            activations, _ = retriever(**batch)
                            pi = torch.clip(torch.softmax(activations, dim=-1), min=1e-8)
                            log_pi = torch.log(pi)

                        # Update entropy coefficient
                        alpha = torch.exp(log_alpha.detach())
                        alpha_loss = (pi * (-log_alpha * (log_pi + target_entropy))).sum(-1).mean()
                        alpha_optim.zero_grad()
                        alpha_loss.backward()
                        alpha_optim.step()

                        # Get q function targets
                        with torch.no_grad():
                            # Get q function estimates for next states, take min to avoid q value explosion
                            if options.sep_val_model:
                                next_q_ests = torch.stack([target(**batch)[0] for target in critic_targets])
                            else:
                                next_q_ests = torch.stack(target(**batch)[1])
                            next_q_ests = torch.min(next_q_ests, dim=0).values
                            soft_val_est = (pi * (next_q_ests - alpha * log_pi)).sum(dim=-1).view(batch_size, -1)
                            # Target is just reward for terminal states; and slice to start at v_1
                            soft_val_est = F.pad(soft_val_est, (0, 1))[:, 1:]
                            q_targets = rewards + soft_val_est

                        # Update critics
                        for critic_idx in range(2):
                            if options.sep_val_model:
                                q_est_vecs = critics[critic_idx](**batch)[0]
                            else:
                                q_est_vecs = retriever(**batch)[1][critic_idx]
                            # Get the estimated q value for each selected action
                            q_ests = torch.gather(
                                q_est_vecs.view(batch_size, max_seq_len, -1),
                                dim=2, index=batch["policy_example_indices"].unsqueeze(2)
                            ).squeeze(2)
                            # Minimize soft Bellman residual and update network
                            loss = F.mse_loss(q_ests, q_targets)
                            critic_optims[critic_idx].zero_grad()
                            loss.backward()
                            if options.sep_val_model:
                                torch.nn.utils.clip_grad_norm_(critics[critic_idx].parameters(), options.grad_clip)
                            else:
                                torch.nn.utils.clip_grad_norm_(retriever.parameters(), options.grad_clip)
                            critic_optims[critic_idx].step()
                            train_stats["vf_losses"].append(loss.item())

                        # Anneal targets towards critics using Polyak update
                        if options.sep_val_model:
                            for critic, target in zip(critics, critic_targets):
                                self.polyak_update(critic, target, options.tau)
                        else:
                            self.polyak_update(retriever, target, options.tau)

                        # Update actor
                        with torch.no_grad():
                            if options.sep_val_model:
                                q_ests = torch.stack([critic(**batch)[0] for critic in critics])
                            else:
                                q_ests = torch.stack(retriever(**batch)[1])
                            q_ests = torch.min(q_ests, dim=0).values
                        activations, _ = retriever(**batch)
                        pi = torch.clip(torch.softmax(activations, dim=-1), min=1e-8)
                        log_pi = torch.log(pi)
                        actor_loss = (pi * (alpha * log_pi - q_ests.detach())).sum(dim=-1).mean()
                        optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(retriever_params, options.grad_clip)
                        optimizer.step()
                        train_stats["losses"].append(actor_loss.item())
                else:
                    raise Exception(f"Algorithm {options.rl_algo} not supported!")

                # Do gradient step for methods without inner loops
                if options.rl_algo not in (RLAlgorithm.DSAC.value, RLAlgorithm.PPO.value):
                    # Maximize entropy - encourages exploration by flattening action distribution
                    entropy = self.get_entropy(activations)
                    entropy = entropy[~ent_mask.view(-1)]
                    loss = loss[~loss_mask.view(-1)]
                    loss = loss.mean() - e_coef * entropy.mean()
                    train_stats["losses"].append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(retriever_params, options.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    if vf_loss is not None:
                        train_stats["vf_losses"].append(vf_loss[~loss_mask.view(-1)].mean().item())

                # If training encoder, re-compute corpus encodings (after each training step)
                if retriever.encoder is not None:
                    dataset.compute_corpus_encodings(show_progress=False)

                # Run validation loop, log stats, and save best model
                if (options.val_every and (batch_idx + 1) % options.val_every == 0) or batch_idx == len(data_loader) - 1:
                    best_val_accuracy = self.validation_loop(
                        retriever, best_model, best_val_accuracy, val_set, val_loader,
                        train_stats, run, options)
                    train_stats = self.reset_train_stats(epoch, cur_lr, supp_reward_anneal)

            # Update learning rate schedulers
            if scheduler is not None:
                scheduler.step()
                if options.rl_algo == RLAlgorithm.DSAC.value:
                    for critic_scheduler in critic_schedulers:
                        critic_scheduler.step()
