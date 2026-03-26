from vllm import LLM, SamplingParams
from typing import List, Optional

class LLMEngine:
    def __init__(
        self, 
        model_name: str, 
        tensor_parallel_size: int = 1,
        is_reasoning_model: bool = False,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9
    ):
        print(f"Loading vLLM engine for {model_name}...")
        self.model_name = model_name
        self.is_reasoning_model = is_reasoning_model
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )

        self.tokenizer = self.llm.get_tokenizer()
        print("✅ vLLM Engine loaded successfully.")

    def generate(
        self, 
        prompts: List[str], 
        temperature: float = 1.0, 
        top_p: float = 0.95, 
        max_tokens: int = 1024,
        stop_sequences: Optional[List[str]] = None,
        n: int = 1  
    ) -> List[List[str]]:
        """
        Generates text for a batch of prompts.
        Returns: [[resp1_prompt1, resp2_prompt1], [resp1_prompt2, resp2_prompt2...]]
        """
        if stop_sequences is None:
            stop_sequences = []

        if self.is_reasoning_model and max_tokens < 2000:
            print("Warning: Reasoning model detected. Bumping max_tokens to 4096.")
            max_tokens = 4096

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_sequences,
            n=n 
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)
        
        results = []
        for output in outputs:
            prompt_responses = [k.text for k in output.outputs]
            results.append(prompt_responses)
            
        return results