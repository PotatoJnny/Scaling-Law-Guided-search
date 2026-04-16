import torch
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig, BitsAndBytesConfig
from .data_structures import State
from .base_reward import BaseRewardEngine


def _apply_armorm_transformers_compat(model_name: str) -> None:
    """Patch small API drifts for ArmoRM remote code on newer transformers."""
    if "ArmoRM-Llama3-8B-v0.1" not in model_name:
        return

    import transformers.models.llama.modeling_llama as llama_modeling

    if not hasattr(llama_modeling, "LLAMA_INPUTS_DOCSTRING"):
        llama_modeling.LLAMA_INPUTS_DOCSTRING = ""

class RMEngine(BaseRewardEngine):
    def __init__(
        self,
        model_name: str,
        quantization: bool = False,
        max_batch_size: int = 64
    ):
        self.model_name = model_name
        self.max_batch_size = max_batch_size

        print(f"Loading Reward Model: {model_name}...")
        _apply_armorm_transformers_compat(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            attn_implementation = "sdpa"
            dtype = None
        else:
            quantization_config = None
            attn_implementation = "sdpa"
            dtype = torch.bfloat16

        # Auto-detect models that use a custom reward model class (e.g. Qwen2ForRewardModel)
        # which register under AutoModel but NOT AutoModelForSequenceClassification
        rm_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        auto_map = getattr(rm_config, 'auto_map', {})
        use_automodel = ('AutoModel' in auto_map and 'AutoModelForSequenceClassification' not in auto_map)

        if use_automodel:
            print(f"  [RMEngine] Detected custom reward model arch ({auto_map.get('AutoModel')}), using AutoModel loader.")
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                num_labels=1,
                trust_remote_code=True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        
        self.tokenizer.truncation_side = 'left'

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval() 
        print("✅ Reward Model loaded successfully.")

    def _strip_leading_bos(self, text: str) -> str:
        bos_token = self.tokenizer.bos_token
        if bos_token and text.startswith(bos_token):
            return text[len(bos_token):]
        return text

    @staticmethod
    def _strip_prompt_suffix(prompt: str, prompt_suffix_to_strip: Optional[str]) -> str:
        if not prompt_suffix_to_strip:
            return prompt

        stripped = prompt.rstrip()
        suffix = prompt_suffix_to_strip.rstrip()
        if stripped.endswith(suffix):
            stripped = stripped[:-len(suffix)].rstrip()
        return stripped

    @staticmethod
    def _split_thinking_and_answer(full_response: str, think_end_token: str = "</think>") -> Tuple[str, str]:
        if think_end_token in full_response:
            thinking, answer = full_response.rsplit(think_end_token, 1)
            return thinking, answer.strip()
        return "", full_response.strip()

    @classmethod
    def _prepare_state_text(
        cls,
        state: State,
        response_mode: Optional[str],
        prompt_suffix_to_strip: Optional[str],
        think_end_token: str,
    ) -> Tuple[str, str]:
        prompt = state.prompt
        response = state.get_full_response()

        if response_mode == "answer_only_after_think":
            prompt = cls._strip_prompt_suffix(prompt, prompt_suffix_to_strip)
            _, answer = cls._split_thinking_and_answer(response, think_end_token=think_end_token)
            if answer:
                response = answer

        return prompt, response

    def score_states_batch(
        self,
        states: List[State],
        rm_instruction: Optional[str] = None,
        response_mode: Optional[str] = None,
        prompt_suffix_to_strip: Optional[str] = None,
        think_end_token: str = "</think>",
    ) -> List[float]:
        """
        Scores multiple states.
        """
        if not states:
            return []

        current_batch_size = min(self.max_batch_size, len(states))
        success = False
        all_scores = []
        
        while not success:
            try:
                for i in range(0, len(states), current_batch_size):
                    batch = states[i : i + current_batch_size]
                    # Pass the instruction down to the internal formatter
                    scores = self._score_internal(
                        batch,
                        rm_instruction,
                        response_mode=response_mode,
                        prompt_suffix_to_strip=prompt_suffix_to_strip,
                        think_end_token=think_end_token,
                    )
                    all_scores.extend(scores)
                success = True
                
            except torch.cuda.OutOfMemoryError:
                # OOM Catcher: Clear VRAM, halve the batch size, and retry
                all_scores = []
                torch.cuda.empty_cache()
                
                if current_batch_size == 1:
                    raise RuntimeError("CUDA OOM in Reward Model even at batch_size=1. Reduce max_length or use quantization.")
                
                current_batch_size = max(1, current_batch_size // 2)
                print(f"⚠️ VRAM Spike in RM! Halving batch size to {current_batch_size} and retrying...")

        return all_scores

    def _score_internal(
        self,
        batch_states: List[State],
        rm_instruction: Optional[str],
        response_mode: Optional[str] = None,
        prompt_suffix_to_strip: Optional[str] = None,
        think_end_token: str = "</think>",
    ) -> List[float]:
        """Internal method to format text and run the forward pass."""
        batch_texts = []
        for state in batch_states:
            chat = []

            prompt_text, response_text = self._prepare_state_text(
                state,
                response_mode=response_mode,
                prompt_suffix_to_strip=prompt_suffix_to_strip,
                think_end_token=think_end_token,
            )
            
            if rm_instruction:
                chat.append({"role": "system", "content": rm_instruction})
                
            chat.extend([
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text}
            ])
            
            formatted_text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Several RM model cards expect chat-templated text without a duplicated BOS.
            formatted_text = self._strip_leading_bos(formatted_text)
            
            if self.tokenizer.eos_token and not formatted_text.strip().endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            
            batch_texts.append(formatted_text)

        with torch.no_grad():
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True, 
                max_length=4096 
            ).to(self.model.device)

            outputs = self.model(**inputs, use_cache=False)
            logits = outputs.logits

            # Different reward models expose logits with slightly different shapes:
            # - standard sequence classifiers: [batch, 1] or [batch, seq_len]
            # - some custom RMs (e.g. ArmoRM): [batch]
            if logits.ndim == 1:
                scores = logits.cpu().tolist()
            else:
                scores = logits[:, -1].cpu().tolist()
            
            if isinstance(scores, float):
                scores = [scores]

            del inputs, outputs, logits
            
        return scores
