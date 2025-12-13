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

        # Determine GPU memory utilization with smart defaults based on model size
        gpu_mem_util = self.options.gpu_memory_utilization
        if gpu_mem_util is None:
            # Auto-adjust based on model name patterns
            model_lower = self.model_name.lower()
            if "0.5b" in model_lower or "500m" in model_lower:
                gpu_mem_util = 0.3  # Small models
            elif "1.5b" in model_lower or "1b" in model_lower:
                gpu_mem_util = 0.4  # Medium models
            elif "2.5b" in model_lower or "2b" in model_lower or "3b" in model_lower:
                gpu_mem_util = 0.5  # Medium-large models
            elif "7b" in model_lower:
                gpu_mem_util = 0.7  # Large models
            elif "13b" in model_lower:
                gpu_mem_util = 0.85  # Very large models
            else:
                gpu_mem_util = 0.5  # Default for unknown models

        print("Loading vLLM generator...")
        print(f"Model: {self.model_name}")
        print(f"max_gen_tokens: {self.options.max_gen_tokens}")
        print(f"GPU memory utilization: {gpu_mem_util}")
        self.llm = LLM(model=self.model_name, gpu_memory_utilization=gpu_mem_util)
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

    def reset_pbar(self, total: int, desc: str):
        """Reset progress bar (no-op for VLLMGenerator as vLLM handles progress internally)."""
        pass

    def generate(self, prompts: List[str], **kwargs) -> List[GeneratorResult]:
        uncached_prompts = [prompt for prompt in prompts if prompt not in self.cache]

        if uncached_prompts:
            # generate LLM responses - vLLM handles everything like batching, stopping, etc.
            results = self.llm.generate(
                uncached_prompts, sampling_params=self.sampling_params
            )
            for prompt, result in zip(uncached_prompts, results):
                generated_text = result.outputs[0].text
                self.cache[prompt] = {"text": generated_text, "nll": 0.0}

            # Save cache periodically
            self._save_cached()
        
        return [self.cache[prompt] for prompt in prompts]


# Global Generator instance for backward compatibility with evaluate.py
# This will be initialized when needed
_generator_instance = None


class Generator:
    """
    Static wrapper class for backward compatibility.
    Wraps the VLLMGenerator instance.
    """
    _instance: VLLMGenerator = None
    
    @classmethod
    def initialize(cls, args: dict):
        """Initialize the global generator instance."""
        cls._instance = VLLMGenerator(args)
        global _generator_instance
        _generator_instance = cls._instance
    
    @classmethod
    def get_instance(cls) -> VLLMGenerator:
        """Get the generator instance."""
        if cls._instance is None:
            raise RuntimeError("Generator not initialized. Call Generator.initialize(args) first.")
        return cls._instance
    
    @classmethod
    def generate(cls, prompts: List[str] = None, **kwargs) -> List[GeneratorResult]:
        """Generate text using the LLM."""
        if cls._instance is None:
            raise RuntimeError("Generator not initialized. Call Generator.initialize(args) first.")
        
        # Handle case where prompts is passed via kwargs
        if prompts is None:
            prompts = kwargs.get('prompts', [])
        
        return cls._instance.generate(prompts, **kwargs)
