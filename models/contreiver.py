import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel

from neural_ir.utils import EncoderOutput

class ContrieverEncoder(PreTrainedModel):
    def __init__(self, model_name='facebook/contriever-msmarco'):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.contriever = AutoModel.from_pretrained(model_name)

    def average_pool(self,last_hidden_states: Tensor,attention_mask: Tensor):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode_query(self,queries):
        q_embs = self.contriever(**queries).last_hidden_state
        q_embs= self.average_pool(q_embs,queries['attention_mask'])
        return q_embs

    def encode_passage(self,passages):
        p_embs = self.contriever(**passages).last_hidden_state
        p_embs= self.average_pool(p_embs,passages['attention_mask'])
        return p_embs
    
    def forward(self,queries=None,passages=None):
        q_embs = self.encode_query(queries) if queries is not None else None

        p_embs = self.encode_passage(passages) if passages is not None else None
        
        return EncoderOutput(q_embs,p_embs)

    @classmethod
    def load(cls,model_name='facebook/contriever-msmarco'):
        model = cls(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model,tokenizer

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

def main():
    model,tokenizer = ContrieverEncoder.load('facebook/contriever-msmarco')

    sentences = [
        "Where was Marie Curie born?",
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(passages=inputs).p_reps

    model.eval()
    with torch.no_grad():
        score01 = embeddings[0] @ embeddings[1] #1.0473
        score02 = embeddings[0] @ embeddings[2] #1.0095

        print(score01,score02)
if __name__=="__main__":
    main()
