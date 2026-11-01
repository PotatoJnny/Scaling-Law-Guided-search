import os
import re
import inspect
from typing import List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def _vllm_supports_engine_arg(name: str) -> bool:
    """Return whether the installed vLLM EngineArgs exposes ``name``."""
    try:
        from vllm.engine.arg_utils import EngineArgs

        return name in inspect.signature(EngineArgs).parameters
    except Exception:
        # Preserve the previous behavior if a future vLLM hides EngineArgs.
        return True


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
_EMPTY_QWEN_THINK_SCAFFOLD_RE = re.compile(
    r"(<\|im_start\|>assistant\s*)<think>\s*</think>\s*$",
    re.DOTALL,
)


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


def _strip_empty_qwen_think_scaffold(text: str) -> str:
    if "<think>" not in text.lower():
        return text
    return _EMPTY_QWEN_THINK_SCAFFOLD_RE.sub(r"\1", text)


class LLMEngine:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        backend: str = "vllm",
        is_reasoning_model: bool = False,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.6,
        enforce_eager: bool = False,
        attention_backend: str = None,
        max_num_batched_tokens: int = None,
        max_batch_size: int = 16,
        max_prompt_batch_size: Optional[int] = None,
        max_samples_per_call: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        async_scheduling: Optional[bool] = None,
        enable_chunked_prefill: Optional[bool] = None,
        enable_prefix_caching: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.is_reasoning_model = is_reasoning_model
        self.max_model_len = max_model_len
        fallback_batch_size = max(1, int(max_batch_size))
        self.max_prompt_batch_size = max(1, int(max_prompt_batch_size or fallback_batch_size))
        self.max_samples_per_call = max(1, int(max_samples_per_call or fallback_batch_size))

        if backend == "transformers":
            print(f"Loading Transformers engine for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()
            self.llm = None
            print("✅ Transformers Engine loaded successfully.")
            return

        print(f"Loading vLLM engine for {model_name}...")

        kwargs = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            disable_custom_all_reduce=tensor_parallel_size > 1,
        )
        supports_async_scheduling = _vllm_supports_engine_arg("async_scheduling")
        if enforce_eager and async_scheduling is None and supports_async_scheduling:
            # Preserve the historical default, but allow explicit runtime-only
            # overrides for retry/stability experiments.
            kwargs["async_scheduling"] = False
        if attention_backend:
            kwargs["attention_backend"] = attention_backend
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = max_num_seqs
        if async_scheduling is not None and supports_async_scheduling:
            kwargs["async_scheduling"] = async_scheduling
        if enable_chunked_prefill is not None:
            kwargs["enable_chunked_prefill"] = enable_chunked_prefill
        if enable_prefix_caching is not None:
            kwargs["enable_prefix_caching"] = enable_prefix_caching
        self.llm = LLM(**kwargs)

        self.tokenizer = self.llm.get_tokenizer()
        print("✅ vLLM Engine loaded successfully.")

    def _chat_template_kwargs(self, output_mode: Optional[str] = None) -> dict:
        kwargs = {}
        if (
            output_mode == "answer_only_no_think"
            and "qwen3" in self.model_name.lower()
        ):
            kwargs["enable_thinking"] = False
        return kwargs

    def render_messages(self, messages: List[dict], output_mode: Optional[str] = None) -> str:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs(output_mode=output_mode),
        )
        if (
            output_mode == "answer_only_no_think"
            and "qwen3" in self.model_name.lower()
        ):
            rendered = _strip_empty_qwen_think_scaffold(rendered)
        return rendered

    def apply_chat_template(
        self,
        user_content: str,
        system_content: str = None,
        output_mode: Optional[str] = None,
    ) -> str:
        """Wrap user_content in the model's chat template.

        Use this when the model is an instruction-tuned chat model and the raw
        prompt text should be presented as a user message (not raw completion).
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return self.render_messages(messages, output_mode=output_mode)

    def generate(
        self, 
        prompts: List[str], 
        temperature: float = 1.0, 
        top_p: float = 0.95, 
        max_tokens: int = 1024,
        stop_sequences: Optional[List[str]] = None,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Generates text for a batch of prompts.
        Returns: [[resp1_prompt1, resp2_prompt1], [resp1_prompt2, resp2_prompt2...]]
        """
        if stop_sequences is None:
            stop_sequences = []

        if self.backend == "transformers":
            results = [[] for _ in prompts]
            do_sample = temperature > 0.0
            for prompt_idx, prompt in enumerate(prompts):
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_model_len - max_tokens,
                )
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                prompt_len = encoded["input_ids"].shape[1]
                for sample_idx in range(n):
                    gen_kwargs = dict(
                        **encoded,
                        max_new_tokens=max_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    if seed is not None:
                        torch.manual_seed(seed + sample_idx)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed + sample_idx)
                    if do_sample:
                        gen_kwargs.update(
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        gen_kwargs.update(do_sample=False)
                    with torch.no_grad():
                        out = self.model.generate(**gen_kwargs)
                    text = self.tokenizer.decode(
                        out[0][prompt_len:],
                        skip_special_tokens=False,
                    )
                    text = _normalize_vllm_output(text)
                    for stop in stop_sequences:
                        if stop and stop in text:
                            text = text.split(stop, 1)[0]
                    results[prompt_idx].append(text)
            return results

        if self.is_reasoning_model and max_tokens < 2000:
            print("Warning: Reasoning model detected. Bumping max_tokens to 4096.")
            max_tokens = 4096

        results = [[] for _ in prompts]

        # Large n or too many prompts at once makes vLLM unstable on our cluster
        # for Phase 1 smoke, so we split generation into smaller sub-calls. We
        # keep prompt batching and per-prompt sample batching separate because
        # their stability limits differ in practice.
        for prompt_start in range(0, len(prompts), self.max_prompt_batch_size):
            prompt_chunk = prompts[prompt_start : prompt_start + self.max_prompt_batch_size]
            for n_start in range(0, n, self.max_samples_per_call):
                n_chunk = min(self.max_samples_per_call, n - n_start)
                chunk_sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    n=n_chunk,
                    seed=None if seed is None else seed + n_start,
                )
                outputs = self.llm.generate(prompt_chunk, chunk_sampling_params, use_tqdm=False)
                for local_idx, output in enumerate(outputs):
                    prompt_responses = [_normalize_vllm_output(k.text) for k in output.outputs]
                    results[prompt_start + local_idx].extend(prompt_responses)

        return results

    def truncate_prompt_to_fit(
        self,
        prompt: str,
        max_output_tokens: int,
        safety_margin: int = 64,
    ) -> str:
        max_input_tokens = max(1, self.max_model_len - max_output_tokens - safety_margin)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= max_input_tokens:
            return prompt
        truncated_ids = token_ids[-max_input_tokens:]
        return self.tokenizer.decode(truncated_ids, skip_special_tokens=False)
