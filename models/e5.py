import torch.nn.functional as F
from torch import nn, Tensor
import torch

from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel

from neural_ir.utils import EncoderOutput

class E5Encoder(PreTrainedModel):
    def __init__(self, model_name='intfloat/e5-base-v2'):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.model = AutoModel.from_pretrained(model_name)

    def average_pool(self,last_hidden_states: Tensor,attention_mask: Tensor):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode_query(self,queries):
        q_embs = self.model(**queries).last_hidden_state
        q_embs= self.average_pool(q_embs,queries['attention_mask'])
        q_embs = F.normalize(q_embs, p=2, dim=1)
        return q_embs

    def encode_passage(self,passages):
        p_embs = self.model(**passages).last_hidden_state
        p_embs= self.average_pool(p_embs,passages['attention_mask'])
        p_embs = F.normalize(p_embs, p=2, dim=1)
        return p_embs
    
    def forward(self,queries=None,passages=None):
        q_embs = self.encode_query(queries) if queries is not None else None

        p_embs = self.encode_passage(passages) if passages is not None else None
        
        return EncoderOutput(q_embs,p_embs)

    @classmethod
    def load(cls,model_name='intfloat/e5-base-v2'):
        model = cls(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model,tokenizer

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

def main():
    model,tokenizer = E5Encoder.load('intfloat/e5-base-v2')
    
if __name__=="__main__":
    main()