from typing import Optional

import torch
from transformers import PreTrainedModel, RobertaConfig, RobertaModel, RobertaTokenizer,AutoModel

class AnceEncoder(PreTrainedModel):
    config_class = RobertaConfig
    TokenizerClass = RobertaTokenizer
    _keys_to_ignore_on_load_missing = [r'position_ids']
    _keys_to_ignore_on_load_unexpected = [r'pooler', r'classifier']

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.embeddingHead = torch.nn.Linear(config.hidden_size, 768)
        self.norm = torch.nn.LayerNorm(768)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.roberta.config.pad_token_id)
            )
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.norm(self.embeddingHead(pooled_output))
        return pooled_output
    
    @classmethod
    def load(cls,model_name="castorini/ance-msmarco-passage"):
        model = cls.from_pretrained(model_name)
        tokenizer = cls.TokenizerClass.from_pretrained(model_name)
        return model,tokenizer

def main():
    model,tokenizer = AnceEncoder.load("castorini/ance-msmarco-passage")
    print(list(model.state_dict().keys()))

if __name__=="__main__":
    main()