import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from .data_structures import State

class RMEngine:
    def __init__(
        self, 
        model_name: str, 
        quantization: bool = False, 
        max_batch_size: int = 64
    ):
        self.model_name = model_name
        self.max_batch_size = max_batch_size

        print(f"Loading Reward Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            attn_implementation = "flash_attention_2"
            dtype = None
        else:
            quantization_config = None
            attn_implementation = "flash_attention_2" if torch.cuda.is_bf16_supported() else None
            dtype = torch.bfloat16


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

    def score_states_batch(self, states: List[State], rm_instruction: Optional[str] = None) -> List[float]:
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
                    scores = self._score_internal(batch, rm_instruction)
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

    def _score_internal(self, batch_states: List[State], rm_instruction: Optional[str]) -> List[float]:
        """Internal method to format text and run the forward pass."""
        batch_texts = []
        for state in batch_states:
            chat = []
            
            if rm_instruction:
                chat.append({"role": "system", "content": rm_instruction})
                
            chat.extend([
                {"role": "user", "content": state.prompt},
                {"role": "assistant", "content": state.get_full_response()}
            ])
            
            formatted_text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            
            if not formatted_text.strip().endswith(self.tokenizer.eos_token):
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

            outputs = self.model(**inputs)
            logits = outputs.logits
            
            scores = logits[:, -1].cpu().tolist()
            
            if isinstance(scores, float):
                scores = [scores]

            del inputs, outputs, logits
            
        return scores