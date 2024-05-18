import torch.nn.functional as F

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import numpy as np
import faiss
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
from neural_ir.utils import EncoderOutput    
from neural_ir.retriever_factory import retreiver_factory
from neural_ir.dataset import EncodeDataset
from neural_ir.collator import EncodeCollator

from contextlib import nullcontext
import pickle
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='') 
    parser.add_argument('--encode_output_path', dest='encode_output_path',default=None,type=str,required=True)
    parser.add_argument('--model_name', dest='model_name',default='facebook/contriever-msmarco',type=str,required=True)
    parser.add_argument('--batch_size', dest='batch_size',default=None,type=int,required=True)
    parser.add_argument('--dataset_name', dest='dataset_name',default=None,type=str,required=True)
    parser.add_argument('--device', dest='device',default='cuda',type=str)
    parser.add_argument('--query_max_len', dest='query_max_len',default=32,type=int)
    parser.add_argument('--passage_max_len', dest='passage_max_len',default=180,type=int)
    parser.add_argument('--encode_is_query', dest='encode_is_query',action='store_true')
    parser.add_argument('--dataset_shard_index', dest='dataset_shard_index',default=0,type=int)
    parser.add_argument('--dataset_number_of_shards', dest='dataset_number_of_shards',default=1,type=int)
    parser.add_argument('--dataloader_num_workers', dest='dataloader_num_workers',default=10,type=int)
    parser.add_argument('--fp16', dest='fp16',action='store_true')
    parser.add_argument('--title', dest='title',action='store_true')
    parser.add_argument('--local_data', dest='local_data',action='store_true')
    parser.add_argument('--use_pseudo_doc', dest='use_pseudo_doc',action='store_true')
    parser.add_argument('--normalize', dest='normalize',action='store_true')
    parser.add_argument('--dataset_split', dest='dataset_split',default=None,type=str)
    parser.add_argument('--dataset_config', dest='dataset_config',default=None,type=str)
    
    args = parser.parse_args()

    encoder,tokenizer = retreiver_factory(args.model_name,normalize=args.normalize)

    encode_dataset = EncodeDataset(
        data_args=args
    )

    encode_collator = EncodeCollator(
        data_args=args,
        tokenizer=tokenizer,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=args.batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
    )

    encoded = []
    lookup_indices = []
    encoder = encoder.to(args.device)
    encoder.eval()

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                if args.encode_is_query:
                    model_output: EncoderOutput = encoder(queries=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = encoder(passages=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded)

    with open(args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

if __name__=='__main__':
    main()