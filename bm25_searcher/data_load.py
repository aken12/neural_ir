from datasets import load_dataset

import argparse
import logging
import json
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def datasets_to_json(args):
    logger.info(args.dataset_name)

    data = load_dataset(
                args.dataset_name,
                args.dataset_config,
                split=args.dataset_split,
            )

    if args.dataset_split == "queries":
        output = f"{args.dataset_name.split('/')[-1]}.tsv"
    else:
        output = f"{args.dataset_name.split('/')[-1]}.jsonl"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir,output)
    
    with open(output_path,'w')as fw:
        for line in tqdm(data):
            if args.dataset_split == "queries":
                fw.write(line['_id'] + '\t' + line['text'] + '\n')
            else:
                json_line = {}
                json_line['id'] = line['_id']
                json_line['title'] = line['title']
                json_line['contents'] = line['text']
                
                json_line = json.dumps(json_line)
                fw.write(json_line)
                fw.write('\n')

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name',help='',required=True)
    parser.add_argument('--dataset_config',help='',default='queries')
    parser.add_argument('--dataset_split',help='',default='queries')
    parser.add_argument('--output_dir',help='',required=True)

    args = parser.parse_args()
    datasets_to_json(args)

if __name__=="__main__":
    main()

