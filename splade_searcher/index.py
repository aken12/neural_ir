from os.path import join as oj
import argparse
from tqdm import tqdm
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader

from neural_ir.splade_searcher.indexing.dataset import IndexDataset
from neural_ir.splade_searcher.indexing.array import IndexDictOfArray
from neural_ir.dataset import EncodeDataset
from neural_ir.collator import EncodeCollator
from neural_ir.models.splade import Splade


def indexing(args):
    model = Splade(model_type_or_dir=args.model_name)
    tokenizer = model.transformer_rep.tokenizer
    model.to(args.device)

    encode_collator = EncodeCollator(data_args=args,tokenizer=tokenizer)

    if args.collection_path != None:
        args.dataset_name = args.collection_path
    encode_dataset = EncodeDataset(data_args=args)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=args.batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
    )

    dim_voc = model.transformer_rep.transformer.config.vocab_size
    sparse_index = IndexDictOfArray(args.index_output_path, dim_voc=dim_voc, force_new=True)    
    count = 0
    doc_ids = []
    logger.info("index process started...")
    with torch.no_grad():
        model.eval()
        for batch in tqdm(encode_loader, desc="Indexing", position=0, leave=True):
            inputs = {k: v.to(args.device) for k, v in batch[1].items()}
            batch_documents = model(d_kwargs=inputs)["d_rep"]
            
            row, col = torch.nonzero(batch_documents, as_tuple=True)
            data = batch_documents[row, col]
            row = row + count

            batch_ids = list(batch[0])
            doc_ids.extend(batch_ids)
            count += len(batch_ids)
            sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                               n_docs=len(batch_ids))

    sparse_index.save()
    pickle.dump(doc_ids, open(oj(args.index_output_path, "doc_ids.pkl"), "wb"))
    logger.info("Done iterating over the corpus!")
    logger.info("index contains {} posting lists".format(len(sparse_index)))
    logger.info("index contains {} documents".format(len(doc_ids)))

def main():
    parser = argparse.ArgumentParser(description='') 
    parser.add_argument('--index_output_path', dest='index_output_path',default=None,type=str,required=True)
    parser.add_argument('--model_name', dest='model_name',default='naver/splade-v3',type=str,required=True)
    parser.add_argument('--batch_size', dest='batch_size',default=None,type=int,required=True)
    parser.add_argument('--dataset_name', dest='dataset_name',default=None,type=str,required=True)
    parser.add_argument('--collection_path', dest='collection_path',default=None,type=str)
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
    parser.add_argument('--lower_text', dest='lower_text',action='store_true')
    # parser.add_argument('--text_normalize', dest='text_normalize',action='store_true')
    
    args = parser.parse_args()
    indexing(args)

if __name__=="__main__":
    main()