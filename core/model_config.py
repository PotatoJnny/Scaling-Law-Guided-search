from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LMConfig:
    model_name: str
    quantization: bool = False
    # temperature: float = 0.3
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 150
    repetition_penalty: float = 1.1
    max_batch_size: Optional[int] = 128
    safety_factor: float = 0.6
    task: str = "math_proof"  
    seed: int = 100
    

@dataclass
class RMConfig:
    model_name: str
    quantization: bool = False
    max_batch_size: Optional[int] = 256
    safety_factor: float = 0.9
    seed : int = 100
    


@dataclass
class SLGConfig:
    K: int = 5                         # Search Width
    m: int = 100                       # Number of samples for estimation
    N: int = 50000                     # Total budget
    max_depth: int = 10                # Maximum states of a response
    verbose: bool = True               # Print progress information
    lm_config: LMConfig = field(default_factory=lambda: LMConfig(model_name="meta-llama/Llama-3-7B-Instruct-v2"))
    rm_config: RMConfig = field(default_factory=lambda: RMConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))




    