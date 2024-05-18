from dataclasses import dataclass
from typing import Dict, Optional
from transformers.file_utils import ModelOutput
from torch import nn, Tensor

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None

