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

    output = f"{args.dataset_name.split('/')[-1]}.tsv"

    dataset_name = f'{args.dataset_name}-qrels'
    data = load_dataset(
                dataset_name,
                args.dataset_config,
                split=args.dataset_split,
            )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir,output)
    
    with open(output_path,'w')as fw:
        for line in tqdm(data):
            fw.write(f"{line['query-id']}\t{'0'}\t{line['corpus-id']}\t{line['score']}\n")

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name',help='',required=True)
    parser.add_argument('--dataset_config',help='',default=None)
    parser.add_argument('--dataset_split',help='',default='test')
    parser.add_argument('--output_dir',help='',required=True)

    args = parser.parse_args()
    datasets_to_json(args)

if __name__=="__main__":
    main()