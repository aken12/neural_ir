from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset 

import random

import logging
import json
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
class EncodeDataset(Dataset):
    def __init__(self,data_args):
        self.data_args = data_args

        logger.info(self.data_args.dataset_name)
        if self.data_args.local_data:
            self.encode_data = {"query_id": [],"query": []} if self.data_args.encode_is_query \
                               else {"docid": [],"text": []} 
            
            if self.data_args.title:
                self.encode_data["title"] = []
            if self.data_args.use_pseudo_doc:
                self.encode_data["pseudo_doc"] = []
            
            with open(self.data_args.dataset_name) as fr:
                if self.data_args.dataset_name.split('.')[-1] == "tsv":
                    for line in fr:
                        line = line.strip().split('\t')

                        if self.data_args.encode_is_query:
                            self.encode_data["query_id"].append(line[0])
                            self.encode_data["query"].append(line[1])
                        else:
                            self.encode_data["docid"].append(line[0])
                            self.encode_data["text"].append(line[1])
                            if self.data_args.title:
                                self.encode_data['title'].append(line[2])
                                
                else:
                    # for line in fr:
                        # line = json.loads(line)
                    json_data = json.load(fr)
                    for line in json_data:
                        if self.data_args.encode_is_query:
                            self.encode_data["query_id"].append(line["query_id"])
                            self.encode_data["query"].append(line["query"])
                            if self.data_args.use_pseudo_doc:
                                self.encode_data["pseudo_doc"].append(line["pseudo_doc"])
                        else:
                            self.encode_data["docid"].append(line["id"])
                            self.encode_data["text"].append(line["contents"])
                            if self.data_args.title:
                                self.encode_data['title'].append(line["title"])

            self.encode_data = HFDataset.from_dict(self.encode_data)

        else:
            
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                split=self.data_args.dataset_split,
            )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    
        logger.info(len(self.encode_data))

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_text = text['query']
            if self.data_args.use_pseudo_doc:
                pseudo_doc = text['pseudo_doc']
                formated_text = formated_text + '[SEP]' + pseudo_doc
                # formated_text = formated_text + '[SEP]' + pseudo_doc
            else:
                formated_text = formated_text
        else:
            text_id = text['docid']
            if self.data_args.title:
                # formated_text = text['title'] + '[SEP]' + text['text']
                # formated_text = "passage: " + text['title'] + ' ' + text['text']
                formated_text = text['title'] + ' ' + text['text']

            else:
                formated_text = text['text']
        return text_id, formated_text
    
class RerankerDataset(Dataset):
    def __init__(self,data_args):
        self.data_args = data_args

        if self.data_args.local_data:
            self.encode_data = {"pairs": [],"labels":[]} 
            
            with open(self.data_args.dataset_name)as fr:
                for line in fr:
                    line = line.strip().split('\t')
                    text = f'質問: {line[0]} 文書: {line[1]} 適合:'
                    self.encode_data["pairs"].append(text)
                    self.encode_data["labels"].append("yes")
                    text = f'質問: {line[0]} 文書: {line[2]} 適合:'
                    self.encode_data["pairs"].append(text)
                    self.encode_data["labels"].append("no")

            self.encode_data = HFDataset.from_dict(self.encode_data)

        else:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                # data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                # cache_dir=self.data_args.dataset_cache_dir,
            )

        # if self.data_args.dataset_number_of_shards > 1:
        #     self.encode_data = self.encode_data.shard(
        #         num_shards=self.data_args.dataset_number_of_shards,
        #         index=self.data_args.dataset_shard_index,
        #     )
    
    def __len__(self):
        return len(self.encode_data)

    # def __getitem__(self, item):
    #     group = self.encode_data[item]

    #     epoch = int(self.trainer.state.epoch)

    #     _hashed_seed = hash(item + self.trainer.args.seed)

    #     query = group['query']
    #     group_positives = group['positives']
    #     group_negatives = group['negatives']
    #     group_bm25_negatives = group['bm25_negatives']
    #     group_original_negatives = group['original_negatives'][:15]
    
    #     pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
    #     passages = []

    #     passages.append(pos_psg)

    #     negative_size = self.data_args.train_group_size - 1
    #     if len(group_negatives) < negative_size:
    #         negs = random.choices(group_negatives, k=negative_size)
    #     elif self.data_args.train_group_size == 1:
    #         negs = []
    #     elif self.data_args.negative_passage_no_shuffle:
    #         negs = group_negatives[:negative_size]
    #     else:
    #         _offset = epoch * negative_size % len(group_negatives)
    #         negs = [x for x in group_negatives]
    #         random.Random(_hashed_seed).shuffle(negs)
    #         negs = negs * 2
    #         negs = negs[_offset: _offset + negative_size]

    #     for neg_psg in negs:
    #         passages.append(self.docid2doc[neg_psg['docid']])
    #         # passages.append(neg_psg['text'])

    #     pos = f"質問: {text['query']} 文書: {text['positive']} 適合:"
    #     pos_labels = "yes"
    #     neg = f"質問: {text['query']} 文書: {text['negative']} 適合:"
    #     neg_labels = "no"
    #     return pos,pos_labels,neg,neg_labels

    def __getitem__(self, item):
        text = self.encode_data[item]
        pos = f"質問: {text['query']} 文書: {text['positive']} 適合:"
        pos_labels = "yes"
        neg = f"質問: {text['query']} 文書: {text['negative']} 適合:"
        neg_labels = "no"
        return pos,pos_labels,neg,neg_labels