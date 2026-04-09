import os
from vllm import LLM, SamplingParams
from typing import List, Optional


def _build_gpt2_byte_decoder():
    """Inverse of the GPT-2 bytes_to_unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_GPT2_BYTE_DECODER = _build_gpt2_byte_decoder()
# Sentinel chars that only appear in GPT2-encoded text, never in normal UTF-8 output
_GPT2_SPACE = "\u0120"   # Ġ — GPT2 encoding of byte 0x20 (space)
_GPT2_NEWLINE = "\u010a"  # Ċ — GPT2 encoding of byte 0x0A (newline)


def _normalize_vllm_output(text: str) -> str:
    """Convert GPT2-style byte-level BPE artifacts back to normal text.

    Some tokenizers (e.g. DeepSeek-Coder) use GPT2's bytes_to_unicode scheme.
    When vLLM decodes their output without applying the inverse mapping, the raw
    unicode surrogates (Ġ for space, Ċ for newline, …) appear verbatim, which
    breaks regex-based code extraction. This function detects and undoes that.
    """
    if _GPT2_SPACE not in text and _GPT2_NEWLINE not in text:
        return text  # fast path: normal text, nothing to fix

    out_bytes = []
    for c in text:
        if c in _GPT2_BYTE_DECODER:
            out_bytes.append(_GPT2_BYTE_DECODER[c])
        else:
            out_bytes.extend(c.encode("utf-8"))
    try:
        return bytes(out_bytes).decode("utf-8", errors="replace")
    except Exception:
        return text


class LLMEngine:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        is_reasoning_model: bool = False,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.6,
        enforce_eager: bool = False,
        attention_backend: str = None,
        max_num_batched_tokens: int = None,
    ):
        print(f"Loading vLLM engine for {model_name}...")
        self.model_name = model_name
        self.is_reasoning_model = is_reasoning_model

        kwargs = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            disable_custom_all_reduce=tensor_parallel_size > 1,
        )
        use_v1_engine = os.environ.get("VLLM_USE_V1") != "0"
        if enforce_eager and use_v1_engine:
            # vLLM 0.19.0 async scheduling + enforce_eager causes
            # cudaErrorIllegalAddress in synchronize_input_prep() on H100
            # and L40S. Disable async scheduling when running in eager mode.
            kwargs["async_scheduling"] = False
        if attention_backend:
            kwargs["attention_backend"] = attention_backend
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        self.llm = LLM(**kwargs)

        self.tokenizer = self.llm.get_tokenizer()
        print("✅ vLLM Engine loaded successfully.")

    def apply_chat_template(self, user_content: str, system_content: str = None) -> str:
        """Wrap user_content in the model's chat template.

        Use this when the model is an instruction-tuned chat model and the raw
        prompt text should be presented as a user message (not raw completion).
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
            prompt_responses = [_normalize_vllm_output(k.text) for k in output.outputs]
            results.append(prompt_responses)
            
        return results
