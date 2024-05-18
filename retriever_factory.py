from models.ance import AnceEncoder
from models.contreiver import ContrieverEncoder
from models.e5 import E5Encoder
from transformers import AutoTokenizer
from models.dense import DenseModel

def retreiver_factory(model_name,normalize=False):
    if model_name == "facebook/contriever-msmarco":
        model,tokenizer = ContrieverEncoder.load(model_name)
    
    elif model_name == "intfloat/e5-base-v2":
        model,tokenizer = E5Encoder.load('intfloat/e5-base-v2')

    elif model_name == "intfloat/e5-large-v2":
        model,tokenizer = E5Encoder.load('intfloat/e5-large-v2')

    elif model_name == "intfloat/e5-base":
        model,tokenizer = E5Encoder.load('intfloat/e5-base')

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = 'right'
        
        model = DenseModel.load(
            model_name,
            pooling="cls",
            normalize=normalize,
        )

    return model,tokenizer