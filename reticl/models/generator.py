import json
import os
from typing import List, TypedDict
from vllm import LLM, SamplingParams
from reticl.utils import TrainOptions

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class GeneratorResult(TypedDict):
    text: str
    nll: float


class VLLMGenerator:
    def __init__(self, args: dict):
        self.options = TrainOptions(args)
        self.model_name = self.options.generator_model
        self.cache_filename = f"generator_cache_{self.options.dataset}_{self.model_name.replace('/', '_')}_ex{self.options.num_examples}_mgt{self.options.max_gen_tokens}.json"
        self.cache = self._get_saved_cache(self.cache_filename)

        print("Loading vLLM generator...")
        print(f"max_gen_tokens: {self.options.max_gen_tokens}")
        self.llm = LLM(model=self.model_name, gpu_memory_utilization=0.3)
        self.sampling_params = SamplingParams(
            max_tokens=self.options.max_gen_tokens, temperature=0.0
        )

    def _get_saved_cache(self, cache_filename: str):
        if os.path.exists(cache_filename):
            with open(cache_filename, encoding="utf-8") as cache_file:
                return json.load(cache_file)
        return {}

    def _save_cached(self):
        temp_cache = self._get_saved_cache(self.cache_filename)
        temp_cache.update(self.cache)
        print(f"Saving cache ({len(temp_cache)} entries) to {self.cache_filename}...")
        with open(self.cache_filename, "w", encoding="utf-8") as cache_file:
            json.dump(temp_cache, cache_file, indent=2, ensure_ascii=False)

    # TODO: clean up logging
    def generate(self, prompts: List[str]):
        uncached_prompts = [prompt for prompt in prompts if prompt not in self.cache]

        # generate LLM responses - vLLM handles everything like batching, stopping, etc.
        results = self.llm.generate(
            uncached_prompts, sampling_params=self.sampling_params
        )
        for prompt, result in zip(uncached_prompts, results):
            generated_text = result.outputs[0].text
            self.cache[prompt] = {"text": generated_text, "nll": 0.0}

        # TODO: periodic cache saving
        self._save_cached()
        return [self.cache[prompt] for prompt in prompts]
