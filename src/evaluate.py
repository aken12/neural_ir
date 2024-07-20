import numpy as np

import json
import argparse
import os 

import pytrec_eval

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_res(run, qrel_file, rel_threshold=1):
    if type(run) == str:
        with open(run, 'r' )as f:
            run_data = f.readlines()
    else:
        run_data=run
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    runs_top10 = {}
    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = 1000 - int(line[3])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

        rank = int(line[3])
        if rank > 10:
            continue
        if query not in runs_top10:
            runs_top10[query] = {}
        runs_top10[query][passage] = rel

    # for qid in run_data:
    #     sorted_passages = sorted(run_data[qid].items(), key=lambda x: x[1], reverse=True)

    #     for rank, (passage, score) in enumerate(sorted_passages, start=1):
    #         if rank > 10:
    #             break
    #         if qid not in runs_top10:
    #             runs_top10[qid] = {}
    #         runs_top10[qid][passage] = score
    # runs = run_data
    
    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.10","recall.100","recall.1000"})
    res = evaluator.evaluate(runs)

    recall_1000_list = [v['recall_1000'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]

    res = evaluator.evaluate(runs_top10)

    mrr_list = [v['recip_rank'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.10"})
    res = evaluator.evaluate(runs)
    ndcg_10_list = [v['ndcg_cut_10'] for v in res.values()]

    res = {
            "MRR@10": round(np.average(mrr_list), 4),
            "NDCG@10": round(np.average(ndcg_10_list), 4),
            "Recall@10": round(np.average(recall_10_list), 4),
            "Recall@100": round(np.average(recall_100_list), 4),
            "Recall@1000": round(np.average(recall_1000_list), 4),
        }
    
    logger.info(res)
    return res

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_path',help='')
    parser.add_argument('--exp',help='',required=True)
    parser.add_argument('--run_dir',help='',default=None)
    parser.add_argument('--run',help='',default=None)
    parser.add_argument('--msmarco',help='',action='store_true')

    args = parser.parse_args()
        
    full_result = {}

    if args.msmarco:
        experiment = args.exp
        qrel_path = f"/home/ace14788tj/extdisk/dense_exp/TAS-B-Reproduction/dataset/msmarco/qrels.{experiment}.small.tsv"
        if args.run_dir:
            for run_file in os.listdir(args.run_dir):
                if (experiment == "fake") and ("cf" not in run_file):
                    continue
                elif experiment not in run_file:
                    continue
                else:
                    full_run_file_path = os.path.join(args.run_dir, run_file)
                    if os.path.isfile(full_run_file_path):
                        res = print_res(run_file=full_run_file_path, qrel_file=qrel_path, rel_threshold=1)
                        full_result[f"{run_file.split('/')[-1].split('.')[0]}"] = res
        with open(os.path.join("results",f'{args.run_dir.split("/")[-3]}_{args.exp}.json'),"w")as fw:
            json.dump(full_result, fw, indent=4)
    
    else:
        experiment = args.exp
        qrel_path = f"/home/ace14788tj/extdisk/neural_ir/bm25_searcher/collections/qrels/{experiment}/{experiment}.tsv"
        res = print_res(run_file=args.run, qrel_file=qrel_path, rel_threshold=1)

if __name__=="__main__":
    main()

    