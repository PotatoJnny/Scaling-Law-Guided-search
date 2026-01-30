from dataclasses import asdict, is_dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoModelForSequenceClassification, BitsAndBytesConfig, AutoConfig
import torch
import json
from core.data_structures import Action, State
import re
from typing import List,Tuple, Optional
import queue
from copy import deepcopy
from torch.multiprocessing import Queue




class StopOnSequence(StoppingCriteria):
    # This can only be used for loop-roll-outs, but cannot be used for batch generation
    def __init__(self, stop_sequences: List[str], tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.min_new_tokens = 0  # Don't stop until we've generated at least 0 new tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Get the original input length from kwargs if available
        if hasattr(self, 'original_length'):
            new_tokens_generated = input_ids.shape[1] - self.original_length
        else:
            # Fallback: assume we need at least some tokens
            new_tokens_generated = input_ids.shape[1]

        # Don't stop too early
        if new_tokens_generated < self.min_new_tokens:
            return False

        # Only check the newly generated portion for stop sequences
        # Decode the last 30 tokens to check for stop patterns
        recent_tokens = input_ids[0][-5:]
        recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)

        # Check for any stop sequence in the recent text
        for stop_seq in self.stop_sequences:
            if stop_seq in recent_text:
                # Make sure it's not just part of the original context
                # Only stop if the sequence appears near the end
                if recent_text.rstrip().endswith(stop_seq.strip()):
                    return True
                # Also check if stop sequence appears after some new content
                if stop_seq in recent_text[-len(stop_seq) * 3:]:  # Check last portion
                    return True

        return False





class DataclassJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder to handle dataclasses."""
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)





def create_prompt_template(question: str, task: str) -> str:
    """
    Formats a dataset example into a standardized prompt.
    
    Args:
        example: A dictionary-like object from the dataset.
                 (e.g., from math_qa, it has a 'Problem' key)

    Returns:
        A formatted prompt string.
    """
    
    if task == "math_proof":
        base_template = '''You are a step-by-step mathematical solution assistant. Generate ONLY ONE logical step at a time.

    Rules:
    - Avoid Thinking in advance, generate response step by step
    - Generate exactly one proof step
    - Use proper mathematical notation
    - Do not generate multiple steps
    - Do not include step numbers
    - If the proof or solution is complete, YOU MUST respond with '***'

    Example:
    Question: Prove that the sum of two even integers is even.
    Step 1: Let m and n be two even integers.
    Step 2: Since m is even, we can write m = 2k for some integer k.
    Step 3: Since n is even, we can write n = 2j for some integer j.
    Step 4: Then, the sum of m and n is m + n = 2k + 2j = 2(k + j), which means m + n is even.
    Step 5: ***
    '''
        
    else:
        base_template = '''You are a helpful and concise AI assistant.
        Provide a direct and complete answer to the user's question. When you have fully answered the question and your response is complete, you MUST end your response with the token '<|EOR|>'. Do not use this token anywhere else.
        '''
        

    # Combine the base template with the new question
    return f"{base_template}\n\nQuestion: {question} \n \nAnswer: "




def _clean_and_post_process(text: str, task: str, current_step_num: int = 0, old_text: Optional[str] = '') -> Action:
    ''' Clean generated text and return Action object'''
    if task == "math_proof":
        cleaned_text = _clean_single_step_for_proof(text, current_step_num)
        if cleaned_text.startswith('***'):
            return Action(
                step_text=f"Step {current_step_num}: That is all for the response.",
                is_final=True
            )
        elif cleaned_text.endswith('***'):
            cleaned_text = cleaned_text[:-3].strip()
            return Action( step_text=f"Step {current_step_num}: " + cleaned_text, is_final=True)
        else:
            return Action(step_text=f"Step {current_step_num}: " + cleaned_text, is_final=False)
    else:
        cleaned_text, is_complete = clean_and_truncate_at_eor(text,old_text)
        return Action(step_text=cleaned_text, is_final=is_complete)


def work_on_last_step(action: Action) -> None:
    """
    Cleans a partial <|EOR|> marker from the end of the
    action's step_text and sets its is_final flag to True.

    This is designed to be called when a generation boundary
    splits the <|EOR|> marker, leaving a prefix of the marker
    at the end of the previous step.
    """
    text = action.step_text
    MARKER = "<|EOR|>" 

    cleaned_text = text


    for i in range(len(MARKER) - 1, 0, -1):
        partial_marker = MARKER[:i] # e.g., MARKER[:3] is "<|E"
        
        if text.endswith(partial_marker):
            # Found a partial marker at the end.
            # Remove it from the text.
            cleaned_text = text[:-i]
            
            # We found the longest possible match,
            # so we can stop checking.
            break
    

    action.step_text = cleaned_text.strip()
    action.is_final = True



def _clean_single_step_for_proof(text: str, current_step_num: int) -> str:
    """Clean generated text"""
    text = text.split('\n', 1)[0]
    
    index_of_triple_star = text.find('***')
    if index_of_triple_star != -1:
        text = text[:index_of_triple_star+3]

    text = text.strip()

    while text.endswith('\n'):
        text = text[:-1]

    if '\n\n' in text:
        text = text.split('\n\n')[0]

    step_pattern = r'\s*step\s+\d+[\s.:]*'
    match = re.search(step_pattern, text, re.IGNORECASE)
    if match:
        step_match = re.search(r'step\s+(\d+)', match.group(), re.IGNORECASE)
        if step_match:
            try:
                step_num = int(step_match.group(1))
                if step_num > current_step_num:
                    text = text[:match.start()].strip()
            except:
                pass

    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    result = '\n'.join(cleaned_lines).strip()

    if result and not result.endswith(('.', '!', '?', ':', ')', '***')):
        if len(result.split()) > 3:
            result += '.'

    return result

def clean_and_truncate_at_eor(text: str, old_text: str) -> Tuple[str, bool]:
    """
    Finds the *first* full EOR marker and truncates the string there.

    Returns:
        A tuple containing:
        - The string before the first marker (if found).
        - A boolean indicating if a marker was detected (True) or not (False).
    """
    MARKER = "<|EOR|>"
    combined_text = old_text + text
    index = combined_text.find(MARKER)
    
    if index != -1:
        # Marker was found.
        # Calculate where the marker starts relative to the *new* text.
        marker_start_in_new_text = index - len(old_text)
        
        if marker_start_in_new_text < 0:
            # Marker starts in old_text. No new text to append.
            return "", True
        else:
            # Marker starts in new_text. Return the part of new_text before it.
            return text[:marker_start_in_new_text].strip(), True
    else:
        # Marker was not found. Return the original new text.
        return text.strip(), False

def _process_job(model, tokenizer, device, state, horizon, batch_size, task, max_new_tokens, temperature, top_p, repetition_penalty, seed) -> List[State]:
    """
    Core job processing logic extracted for cleaner worker loop.
    Processes a batch of rollouts for a given state and horizon.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    active_states = [deepcopy(state) for _ in range(batch_size)]
    active_indices = list(range(batch_size))
    completed_states = [None] * batch_size


    for step_idx in range(horizon):
        if not active_indices:
            break

        step_num = len(active_states[0].steps) + 1

        if task == "math_proof":
            batch_contexts = [
                s.get_full_text() + f"\nStep {step_num}:"
                for s in active_states
            ]
        else:
            batch_contexts = [
                s.get_full_text()
                for s in active_states
            ]

        inputs = tokenizer(
            batch_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(device)

        original_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": tokenizer.pad_token_id,
                "repetition_penalty": repetition_penalty,
                "num_return_sequences": 1,
            }

            if task != "math_proof":
                gen_kwargs["min_new_tokens"] = max_new_tokens

            outputs = model.generate(**inputs, **gen_kwargs)

        new_texts = tokenizer.batch_decode(
            outputs[:, original_length:],
            skip_special_tokens=True
        )

        old_texts = tokenizer.batch_decode(
            outputs[:, original_length - 7:original_length],
            skip_special_tokens=True
        )

        del outputs, inputs

        next_active_states = []
        next_active_indices = []

        for i, (orig_idx, s) in enumerate(zip(active_indices, active_states)):
            #print("original text:", new_texts[i])
            new_action = _clean_and_post_process(new_texts[i], task, step_num, old_texts[i])
            if new_action.is_final and new_action.step_text == '':
                work_on_last_step(s.steps[-1])
                is_complete = True
            else:
                s.append_step(new_action)
                is_complete = new_action.is_final

            if is_complete:
                s.is_complete = True
                completed_states[orig_idx] = s
            else:
                next_active_states.append(s)
                next_active_indices.append(orig_idx)

        active_states = next_active_states
        active_indices = next_active_indices

    for orig_idx, s in zip(active_indices, active_states):
        s.is_complete = True
        completed_states[orig_idx] = s

    del active_states, active_indices
    torch.cuda.empty_cache()

    return completed_states


def _worker_loop(
    model_name: str,
    quantization: bool,
    task: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    gpu_id: int,
    work_queue: Queue,
    result_queue: Queue,
    status_queue: Queue,
    seed: int = 42
):
    """
    A loop that runs on a single GPU, waiting for tasks.
    It loads the model ONCE, then processes jobs.
    """

    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        torch.manual_seed(seed + gpu_id)
        torch.cuda.manual_seed_all(seed + gpu_id)

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
            attn_implementation = None
            dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            dtype=dtype,
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[GPU {gpu_id}] Worker ready.")
        status_queue.put(("ready", gpu_id))

        jobs_completed = 0

        while True:
            try:
                job = work_queue.get(timeout = 0.1)

                if job == "STOP":
                    print(f"[GPU {gpu_id}] Received STOP signal. Completed {jobs_completed} jobs.")
                    status_queue.put(("stopped", gpu_id))
                    break

                job_id, state, horizon, batch_size = job
                status_queue.put(("working", gpu_id, job_id))
                current_batch_size = batch_size

                success = False
                while not success and current_batch_size >= 1:
                    try:
                        completed_states = _process_job(
                            model, tokenizer, device,
                            state, horizon, current_batch_size,
                            task, max_new_tokens,
                            temperature, top_p,
                            repetition_penalty, job_id + seed
                        )
                        result_queue.put((gpu_id, job_id, completed_states))
                        jobs_completed += 1
                        status_queue.put(("idle", gpu_id))
                        success = True


                    except torch.cuda.OutOfMemoryError:
                        print(f"[GPU {gpu_id}] OOM with batch size {current_batch_size}. Reducing batch size.")
                        torch.cuda.empty_cache()
                        current_batch_size = current_batch_size // 2
                        if current_batch_size < 1:
                            print(f"[GPU {gpu_id}] Cannot process even batch size 1 due to OOM.")
                            result_queue.put((gpu_id, job_id, []))
                            status_queue.put(("idle", gpu_id))
                            success = True

                    except Exception as e:
                        print(f"[GPU {gpu_id}] Error during job processing: {e}")
                        result_queue.put((gpu_id, job_id, []))
                        status_queue.put(("idle", gpu_id))
                        success = True

            except queue.Empty:
                continue

                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error in worker loop: {e}")
                result_queue.put((gpu_id, [], 0.0))
                break
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error in worker setup: {e}")
        status_queue.put(("error", gpu_id, str(e)))


