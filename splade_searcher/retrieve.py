import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append('..')
sys.path.append('.')
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


from neural_ir.utils import json_dumps_arguments,check_dir_exist_or_build
from neural_ir.src.evaluate import print_res
from neural_ir.dataset import EncodeDataset
from neural_ir.collator import EncodeCollator
from neural_ir.models.splade import Splade
from neural_ir.splade_searcher.indexing.sparse_indexer import SparseRetrieval

def sparse_retrieve_and_evaluate(args):
    logger.info("実行")
    model = Splade(model_type_or_dir=args.model_name)
    tokenizer = model.transformer_rep.tokenizer
    model.to(args.device)
    
    encode_collator = EncodeCollator(data_args=args,tokenizer=tokenizer)

    encode_dataset = EncodeDataset(data_args=args)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=args.batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
    )

    # get query embeddings
    qid2emb = {}
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm(encode_loader, desc="Inferencing"):
            inputs = {k: v.to(args.device) for k, v in batch[1].items()}
            # batch_query_embs, _ = model(**inputs)["q_rep"]
            batch_query_embs = model.encode(inputs,is_q=True)
            qids = batch[0] 
            for i, qid in enumerate(qids):
                qid2emb[qid] = batch_query_embs[i]
    
    # retrieve
    dim_voc = model.output_dim
    retriever = SparseRetrieval(args.index_dir_path, args.retrieval_output_path, dim_voc, args.top_n)
    result = retriever.retrieve(qid2emb)
    
    # evaluate
    eval_kwargs = {"run_file":result, 
                   "qrel_file": args.qrel_file, 
                   "rel_threshold": 1}
    print_res(**eval_kwargs)

    logger.info("Evaluation OK!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='') 
    parser.add_argument('--index_dir_path', dest='index_dir_path',default=None,type=str,required=True)
    parser.add_argument('--model_name', dest='model_name',default='naver/splade-v3',type=str,required=True)
    parser.add_argument('--batch_size', dest='batch_size',default=None,type=int,required=True)
    parser.add_argument('--dataset_name', dest='dataset_name',default=None,type=str,required=True)
    parser.add_argument('--device', dest='device',default='cuda',type=str)
    parser.add_argument('--qrel_file', dest='qrel_file',type=str)
    parser.add_argument('--query_max_len', dest='query_max_len',default=32,type=int)
    parser.add_argument('--passage_max_len', dest='passage_max_len',default=180,type=int)
    parser.add_argument('--fp16', dest='fp16',action='store_true')
    parser.add_argument('--encode_is_query', dest='encode_is_query',action='store_true')
    parser.add_argument('--local_data', dest='local_data',action='store_true')
    parser.add_argument('--title', dest='title',action='store_true')
    parser.add_argument('--use_pseudo_doc', dest='use_pseudo_doc',action='store_true')
    parser.add_argument('--dataset_split', dest='dataset_split',default=None,type=str)
    parser.add_argument('--dataset_config', dest='dataset_config',default=None,type=str)
    parser.add_argument('--dataset_shard_index', dest='dataset_shard_index',default=0,type=int)
    parser.add_argument('--dataset_number_of_shards', dest='dataset_number_of_shards',default=1,type=int)
    parser.add_argument('--dataloader_num_workers', dest='dataloader_num_workers',default=10,type=int)

    # test input file
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")
    
    # test parameters 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--top_n", type=int, default=100)

    # output file
    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")

    args = parser.parse_args()

    # check_dir_exist_or_build([args.retrieval_output_path], force_emptying=args.force_emptying_dir)
    json_dumps_arguments(os.path.join(args.retrieval_output_path, "parameters.txt"), args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    sparse_retrieve_and_evaluate(args)
