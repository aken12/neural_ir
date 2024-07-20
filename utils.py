import os
from dataclasses import dataclass
from typing import Dict, Optional
import json
import shutil

from transformers.file_utils import ModelOutput
from torch import nn, Tensor

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None

def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

def check_dir_exist_or_build(dir_list, force_emptying:bool = False):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
        elif len(os.listdir(x)) > 0:    # not empty
            if force_emptying:
                print("Forcing to erase all contens of {}".format(x))
                shutil.rmtree(x)
                os.makedirs(x)
            else:
                raise FileExistsError
        else:
            continue

def json_dumps_arguments(output_path, args):   
    with open(output_path, "w") as f:
        params = vars(args)
        if "device" in params:
            params["device"] = str(params["device"])
        f.write(json.dumps(params, indent=4))