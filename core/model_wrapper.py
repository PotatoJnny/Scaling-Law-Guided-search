import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, AutoConfig
from typing import List, Tuple
from core.model_config import LMConfig, RMConfig
from core.data_structures import State, Action
from copy import deepcopy
import time
import tqdm
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from core.tools import _clean_and_post_process,work_on_last_step, _worker_loop 


class RMWrapper:
    def __init__(self, RM_config: RMConfig):
        model_name = RM_config.model_name
        quantization = RM_config.quantization
        self.safety_factor = RM_config.safety_factor
        self.max_batch_size = RM_config.max_batch_size

        print(f"Loading reward model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval() 
        print("✅ Reward model loaded successfully.")
        
    def score_state(self, state: State) -> float:
        """Calculate reward for a single state."""
        if not state.is_complete and (not state.steps or not state.steps[-1].is_final):
            raise ValueError("Can only score completed states.")

        scores = self.score_states_batch([state])
        return scores[0]


    def score_states_batch(self, states: List[State]) -> List[float]:
        """
        Score multiple states with adaptive batching to prevent OOM.
        
        Args:
            states: List of State objects to score
            max_batch_size: Maximum batch size (auto-calculated if None)
        
        Returns:
            List of scores in the same order as input states
        """
        if not states:
            return []

        max_batch_size = self.calculate_batch_size(states[0])
        max_batch_size = min(max_batch_size, len(states))
        print(f"Using RM batch size: {max_batch_size}")

        success = False
        all_scores = []
        
        while not success and max_batch_size > 1:
            try:
                for batch_start in range(0, len(states), max_batch_size):
                    batch_end = min(batch_start + max_batch_size, len(states))
                    batch_states = states[batch_start:batch_end]
                    
                    batch_scores = self._score_batch_internal(batch_states)
                    all_scores.extend(batch_scores)
                    
                    if torch.cuda.is_available():
                        del batch_states
                        del batch_scores
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    time.sleep(0.1)
                success = True
            except torch.cuda.OutOfMemoryError:
                all_scores = []
                torch.cuda.empty_cache()
                max_batch_size = max(1, max_batch_size // 2)
        return all_scores
    

    def calculate_batch_size(self, sample_state: State) -> int:
        if torch.cuda.device_count() == 0:
            return 1
        
        safety_factor = self.safety_factor
        gpu_used_set = set(self.model.hf_device_map.values())

        free_memory_list = [torch.cuda.mem_get_info(i)[0] / 1e9 for i in gpu_used_set]
        bottleneck_gpu_id = free_memory_list.index(min(free_memory_list))
        available_memory_gb = free_memory_list[bottleneck_gpu_id] 

        with torch.cuda.device(bottleneck_gpu_id):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            initial_used_memory_gb = torch.cuda.memory_allocated() / 1e9
            _ = self._score_batch_internal([sample_state])
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
            memory_per_item_gb = peak_memory_gb - initial_used_memory_gb
            torch.cuda.empty_cache()

        if memory_per_item_gb <= 0.01:
            print(f"⚠️  Could not reliably estimate memory per item ({memory_per_item_gb:.3f} GB). Defaulting to batch size 100.")
            return 100
        
        print(f"Empirically measured memory per item: {memory_per_item_gb:.2f} GB")
        memory_budget_gb = available_memory_gb * safety_factor
        batch_size = max(1, int(memory_budget_gb / memory_per_item_gb))
        print(f"Calculated batch size: {batch_size}")
        return min(batch_size, self.max_batch_size)
        


    def estimate_memory_per_reward(self, sample_state: State) -> int:
        """
        Estimate optimal batch size based on available GPU memory.
        """

        free_memory, total_memory = torch.cuda.mem_get_info()
        

        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            free_before, _ = torch.cuda.mem_get_info()
            _ = self._score_batch_internal([sample_state])
            free_after, _ = torch.cuda.mem_get_info()
            
            memory_per_item = free_before - free_after            

            torch.cuda.empty_cache()
            
            if memory_per_item > 0:
                return memory_per_item
            
        except Exception as e:
            print(f"Could not estimate RM batch size: {e}")
            return float('inf')
        finally:
            torch.cuda.empty_cache()



    def _score_batch_internal(self, batch_states: List[State]) -> List[float]:
        """
        Internal method to score a single batch without subdividing.
        """
        batch_texts = []
        for state in batch_states:
            prompt = state.prompt
            response = state.get_full_response()
            chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            formatted_text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            
            if not formatted_text.strip().endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            
            batch_texts.append(formatted_text)


        with torch.no_grad():
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            outputs = self.model(**inputs)
            logits = outputs.logits
            
            scores = logits[:, -1].cpu().tolist()

            
            if isinstance(scores, float):
                scores = [scores]

            del inputs
            del outputs
            del logits
        
        return scores


class LLMWrapper:
    def __init__(self, LM_config: LMConfig):
        """
        Initializes the wrapper, adaptively choosing a parallelism strategy.
        """

        self.model_name = LM_config.model_name
        self.quantization = LM_config.quantization
        self.safety_factor = LM_config.safety_factor
        self.max_batch_size = LM_config.max_batch_size
        self.temperature = LM_config.temperature
        self.top_p = LM_config.top_p
        self.max_new_tokens = LM_config.max_new_tokens
        self.repetition_penalty = LM_config.repetition_penalty
        self.task = LM_config.task
        self.seed = LM_config.seed
        self.num_process = 0

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
           torch.cuda.manual_seed_all(self.seed)

        
        print(f"--- Initializing LLMWrapper for {self.model_name} ---")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.attn_implementation = "flash_attention_2"
            self.dtype = None 
        else:
            self.quantization_config = None
            self.attn_implementation = None
            self.dtype = torch.bfloat16
        
        max_memory_map = self._get_memory_map()


        if self.parallel == "data_parallel":
            
            print("   Mode: Data Parallelism across GPUs.")
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            self.task_queues = [Queue() for _ in range(self.num_gpus)]
            self.result_queue = Queue()
            self.status_queue = Queue()
            self.workers = []
            self.active_gpus = list(range(self.num_gpus))

            for gpu_id in self.active_gpus:
                worker_args = (
                    self.model_name, self.quantization, self.task,
                    self.max_new_tokens, self.temperature, self.top_p,
                    self.repetition_penalty,
                    gpu_id,
                    self.task_queues[gpu_id],
                    self.result_queue,
                    self.status_queue,
                    self.seed
                )
                process = Process(target=_worker_loop, args=worker_args)
                process.start()
                self.workers.append(process)
            
            print("✅ LLM Worker processes started.")

            ready_gpus = set()
            while len(ready_gpus) < len(self.active_gpus):
                status, gpu_id, *rest = self.status_queue.get()
                if status == "ready":
                    ready_gpus.add(gpu_id)
                    print(f"   [GPU {gpu_id}] is ready.")
                elif status == "error":
                    raise RuntimeError(f"Error initializing GPU {gpu_id}: {rest[0]}")
                
        
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.quantization_config,
                device_map="auto",
                max_memory=max_memory_map, 
                attn_implementation=self.attn_implementation,
                torch_dtype=self.dtype
            )
            self.model.eval()
            print("✅ Model loaded successfully.")
            
            self.task_queues = None
            self.result_queue = None
            self.status_queue = None
            self.workers = None
            self.active_gpus = [0] # Main process uses GPU 0 or device_map
        
    def __del__(self):
        """Ensure the worker processes are closed."""
        if self.workers:
            print("Shutting down LLM worker processes...")
            for q in self.task_queues:
                q.put("STOP")
                q.close()
                q.join_thread()
            
            self.status_queue.close()
            self.status_queue.join_thread()
            self.result_queue.close()
            self.result_queue.join_thread()

            for p in self.workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            print("LLM Workers shut down.")


    def _get_memory_map(self) -> dict | None:
        """
        Estimates model size and returns a max_memory map if the model
        can fit on a single GPU.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            self.parallel = "single_gpu"
            return None

        print("🔍 Performing adaptive memory check...")
        model_size_bytes = self.estimate_model_size_bytes()
        gpu0_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        safety_margin = 0.4

        print(f"   Model size: ~{model_size_bytes / 1e9:.2f} GB. GPU 0 capacity: {gpu0_memory_bytes / 1e9:.2f} GB.")

        if model_size_bytes < (gpu0_memory_bytes * safety_margin):
            print("   ✅ Model fits on a single GPU. Forcing to GPU 0 for Data Parallelism.")
            max_memory_map = {i: 0 for i in range(num_gpus)}
            max_memory_map[0] = int(gpu0_memory_bytes)
            self.parallel = "data_parallel"
            return max_memory_map
        else:
            print("   ⚠️  Model likely too large. Allowing default Model Parallelism.")
            self.parallel = "model_parallel"
            return None 

    def estimate_model_size_bytes(self) -> int:
        """Estimates the model's size in bytes using its config."""
        print("   Estimating model size from config...")
        try:
            config = AutoConfig.from_pretrained(self.model_name)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)

            total_params = sum(p.numel() for p in model.parameters())
            
            if self.dtype in (torch.float16, torch.bfloat16):
                bytes_per_param = 2
            elif self.dtype == torch.float32:
                bytes_per_param = 4
            else: 
                bytes_per_param = 2 

            return total_params * bytes_per_param
        except Exception as e:
            print(f"   Could not estimate model size: {e}. Defaulting to model parallel.")
            return float('inf')



    def calculate_batch_size(self, state: State, horizon: int, total_rollout_num: int) -> Tuple[int, List]:
        if self.num_gpus == 0:
            return 1, []



        if self.parallel == "data_parallel":
            return self.max_batch_size, []
        
        if self.num_gpus == 1:
            return self.max_batch_size, []

        safety_factor = self.safety_factor
        free_memory_list = [torch.cuda.mem_get_info(i)[0] / 1e9 for i in range(self.num_gpus)]
        available_memory_gb = min(free_memory_list)
        bottleneck_gpu_id = free_memory_list.index(available_memory_gb)
        
        print(f"Using bottleneck GPU {bottleneck_gpu_id} with {available_memory_gb:.2f} GB free")

        memory_per_item_gb = 0
        try:
            with torch.cuda.device(bottleneck_gpu_id):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                initial_used_memory_gb = torch.cuda.memory_allocated() / 1e9


                results = self.perform_n_rollouts(state, horizon, 1)
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                memory_per_item_gb = peak_memory_gb - initial_used_memory_gb
                
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error during memory estimation dry run: {e}")
            return self.max_batch_size, []

        if memory_per_item_gb <= 0.01: 
            print(f"⚠️  Could not reliably estimate memory per item ({memory_per_item_gb:.3f} GB). Defaulting to batch size 1.")
            return self.max_batch_size, []

        print(f"Empirically measured memory per item: {memory_per_item_gb:.2f} GB")

        memory_budget_gb = available_memory_gb * safety_factor
        

        batch_size = max(1, int(memory_budget_gb / memory_per_item_gb))
        print(f"Calculated batch size: {batch_size}")

        return min(batch_size, self.max_batch_size), results 



    def _perform_n_rollouts_data_parallel(self, state: State, horizon: int, n: int) -> List[State]:
        """
        Performs n INDEPENDENT rollouts in parallel batches using DataParallel.
        This version dynamically dispatches jobs and correctly handles shortfalls
        from worker-side OOM errors.
        """
        
        num_workers = len(self.active_gpus)
        print(f"\n Data parallel: Distributing {n} rollouts across {num_workers} GPUs.")

        # Get initial batch size and any pre-calculated results
        batch_size, all_completed_states = self.calculate_batch_size(state, horizon, n)

        # --- Handle edge cases ---
        if len(all_completed_states) >= n:
            return all_completed_states[:n]
        if batch_size <= 0:
            print(f"Warning: Batch size is 0 but still need {n - len(all_completed_states)} rollouts. Returning what we have.")
            return all_completed_states[:n]

        # --- State tracking (tracks rollouts, not jobs) ---
        num_rollouts_completed = len(all_completed_states)
        num_rollouts_dispatched = len(all_completed_states) # Tracks rollouts *requested*
        
        # Map: job_id -> (gpu_id, requested_batch_size)
        active_jobs = {} 
        available_gpus = list(self.active_gpus)

        # --- 1. Initial Job Dispatch ---
        while available_gpus and num_rollouts_dispatched < n:
            # Determine batch size for this new job
            current_batch_size = min(batch_size, n - num_rollouts_dispatched)
            if current_batch_size <= 0:
                break 

            gpu_id = available_gpus.pop(0)
            
            # Use your original job_id logic
            job_id = self.num_process * (len(self.active_gpus) + 1)
            self.num_process += 1

            job = (job_id, state, horizon, current_batch_size)
            self.task_queues[gpu_id].put(job)
            
            num_rollouts_dispatched += current_batch_size
            active_jobs[job_id] = (gpu_id, current_batch_size)
            

        while num_rollouts_completed < n:
            if not active_jobs:
                print(f"Error: Rollout target {n} not met ({num_rollouts_completed}), but no jobs are active.")
                break 


            res_gpu_id, res_job_id, completed_states = self.result_queue.get()

            if res_job_id not in active_jobs:
                print(f"Warning: Received result for unknown job_id {res_job_id}. Ignoring.")
                continue

            _, requested_batch_size = active_jobs.pop(res_job_id)
            
            actual_completed_count = len(completed_states)
            all_completed_states.extend(completed_states)
            num_rollouts_completed += actual_completed_count
            del completed_states 


            shortfall = requested_batch_size - actual_completed_count
            if shortfall > 0:
                num_rollouts_dispatched -= shortfall


            if num_rollouts_dispatched < n:
                current_batch_size = min(batch_size, n - num_rollouts_dispatched)
                
                if current_batch_size > 0:
                    job_id = self.num_process * (len(self.active_gpus) + 1)
                    self.num_process += 1

                    job = (job_id, state, horizon, current_batch_size)
                    self.task_queues[res_gpu_id].put(job) 
                    
                    num_rollouts_dispatched += current_batch_size
                    active_jobs[job_id] = (res_gpu_id, current_batch_size)
                else:
                    available_gpus.append(res_gpu_id)
            else:
                available_gpus.append(res_gpu_id)

        while active_jobs:
            res_gpu_id, res_job_id, completed_states = self.result_queue.get()
            if res_job_id in active_jobs:
                active_jobs.pop(res_job_id)
                del completed_states
            else:
                print(f"Warning: Draining unknown job {res_job_id}")

        return all_completed_states[:n]


        
    def perform_n_rollouts(self, state: State, horizon: int, n: int) -> List[State]:
        """
        Performs n INDEPENDENT rollouts in parallel batches.
        Chooses strategy based on model parallelism mode.
        """
        if self.parallel == "data_parallel":
            return self._perform_n_rollouts_data_parallel(state, horizon, n)
        else:
            batch_size, results = self.calculate_batch_size(state, horizon, n)
            batch_size = min(batch_size, n - len(results))
            if batch_size <= 0:
                return results[:n]
            completed_states = results
            for i in tqdm.tqdm(range(0, n - len(results), batch_size), desc="Rollout Batches"):
                current_batch_size = min(batch_size, n - i)
                batch_completed = self._generate_batch_rollouts_parallel(state, horizon, current_batch_size)
                self.num_process += 1
                completed_states.extend(batch_completed)
                del batch_completed
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            return completed_states


    def _generate_batch_rollouts_parallel(self, state: State, horizon: int, batch_size: int) -> List[State]:
        """
        Generate batch_size complete rollouts in parallel.
        All rollouts in this batch progress together step-by-step.
        """

        torch.manual_seed(self.seed + self.num_process)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + self.num_process)

        active_states = [deepcopy(state) for _ in range(batch_size)]
        active_indices = list(range(batch_size))
        completed_states = [None] * batch_size
        
        for step_idx in range(horizon):
            if not active_indices:
                break

            step_num = len(active_states[0].steps) + 1
            batch_contexts = [
                s.get_full_text() + f"\nStep {step_num}:"
                for s in active_states
            ]
            

            inputs = self.tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            
            original_length = inputs.input_ids.shape[1]
            
            with torch.no_grad(): 
                if self.task == "math_proof":
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=self.repetition_penalty,
                        num_return_sequences=1,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        min_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=self.repetition_penalty,
                        num_return_sequences=1,
                    )



            old_texts = self.tokenizer.batch_decode(
                outputs[:, original_length - 7:original_length],
                skip_special_tokens=True
            )
            
            new_texts = self.tokenizer.batch_decode(
                outputs[:, original_length:],
                skip_special_tokens=True
            )
            
            del outputs
            del inputs


            # Process results and update states
            next_active_states = []
            next_active_indices = []
            
            for i, (orig_idx, s) in enumerate(zip(active_indices, active_states)):
                new_action = _clean_and_post_process(new_texts[i], self.task, step_num, old_texts[i])
                if new_action.is_final and new_action.step_text == '':
                    work_on_last_step(s.steps[-1])
                    is_complete = True
                else:
                    s.append_step(new_action)
                    is_complete = new_action.is_final

                # Move to completed or keep active
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
        
        return completed_states


   
