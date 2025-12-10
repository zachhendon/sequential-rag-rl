import json
import os
from typing import List, TypedDict, Optional
from vllm import LLM, SamplingParams
from reticl.utils import TrainOptions
import time
from tqdm import tqdm

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class GeneratorResult(TypedDict):
    text: str
    nll: float


class VLLMGenerator:
    def __init__(self, args: dict):
        self.options = TrainOptions(args)
        self.model_name = self.options.generator_model

        # setup in-memory and on-disk caches
        self.cache_filename = f"generator_cache_{self.options.dataset}_{self.model_name.replace('/', '_')}_ex{self.options.num_examples}_mgt{self.options.max_gen_tokens}.jsonl"
        self.memory_cache = self._load_disk_cache(self.cache_filename)
        self.pbar = None
        self.cached_count = 0
        self.uncached_count = 0

        print("Loading vLLM generator...")
        print(f"max_gen_tokens: {self.options.max_gen_tokens}")
        self.llm = LLM(model=self.model_name, gpu_memory_utilization=0.3)
        self.sampling_params = SamplingParams(
            max_tokens=self.options.max_gen_tokens, temperature=0.0
        )

    def _load_disk_cache(self, cache_filename: str):
        # Return an empty dictionary if the cache file doesn't exist
        cache = {}
        if not os.path.exists(cache_filename):
            return cache

        # Load the cache into a python dictionary if the file exists
        with open(cache_filename, "r", encoding="utf-8") as cache_file:
            for line in cache_file:
                entry = json.loads(line.strip())
                prompt = entry["prompt"]
                cache[prompt] = {"text": entry["text"], "nll": entry["nll"]}
        print(f"Loaded {len(cache)} entries from cache")
        return cache

    def _update_disk_cache(self, new_cache_items: dict):
        # Append new cache entries to the JSONL file
        with open(self.cache_filename, "a", encoding="utf-8") as cache_file:
            for prompt, value in new_cache_items.items():
                entry = {"prompt": prompt, "text": value["text"], "nll": value["nll"]}
                cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def reset_pbar(self, train_size: int, desc: str = "Generating"):
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=train_size, desc=desc, unit=" samples", leave=True)
        self.cached_count = 0
        self.uncached_count = 0
        self._update_pbar_description()

    def _update_pbar_description(self):
        base_desc = self.pbar.desc.split(' (')[0]  # Get base description without cache info
        if self.cached_count > 0 and self.uncached_count > 0:
            self.pbar.set_description(f"{base_desc} (cached: {self.cached_count}, generated: {self.uncached_count})")
        elif self.cached_count > 0:
            self.pbar.set_description(f"{base_desc} (all {self.cached_count} cached)")
        elif self.uncached_count > 0:
            self.pbar.set_description(f"{base_desc} (all {self.uncached_count} generated)")

    def generate(self, prompts: List[str]):
        uncached_prompts = [
            prompt for prompt in prompts if prompt not in self.memory_cache
        ]
        new_cache_items = {}

        # generate LLM responses - vLLM handles everything like batching, stopping, etc.
        if uncached_prompts:
            results = self.llm.generate(
                uncached_prompts, sampling_params=self.sampling_params, use_tqdm=False
            )
            for prompt, result in zip(uncached_prompts, results):
                generated_text = result.outputs[0].text
                new_cache_items[prompt] = {"text": generated_text, "nll": 0.0}

            # update in-memory and on-disk caches
            self.memory_cache.update(new_cache_items)
            self._update_disk_cache(new_cache_items)

        # Update progress bar
        self.pbar.update(len(prompts))
        self.uncached_count += len(uncached_prompts)
        self.cached_count += len(prompts) - len(uncached_prompts)
        self._update_pbar_description()

        return [self.memory_cache[prompt] for prompt in prompts]
