import numpy as np

import os
import json
import argparse
from tqdm import tqdm,trange
# from pyserini.search.lucene import LuceneSearcher
from pyserini.search import SimpleSearcher
import pytrec_eval

import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def bm25(args):
    # bm25_k1 = 0.82
    # bm25_b = 0.68
    bm25_k1 = 0.9
    bm25_b = 0.4
    topk = 1000

    query_list = []
    qid_list = []

    # with open(args.qrel_path)as fr:
    #     for line in fr:
    #         line = line.strip().split('\t')
    #         query = line[0]
    #         passage = line[2]
    #         rel = int(line[3])
    #         if query not in qrels:
    #             qrels[query] = {}
    #         qrels[query][passage] = rel

    # logger.info(f'{len(qrels)=}')

    with open(args.query_path)as fr:
        for query_line in fr:
            query_line = query_line.strip().split('\t')
            query = query_line[1]

            qid = query_line[0]

            query_list.append(query)
            qid_list.append(qid)

    logger.info(f'{len(qid_list)=}')

    searcher = SimpleSearcher(args.index_dir)
    searcher.set_bm25(bm25_k1, bm25_b)
    logger.info("start search...")

    os.makedirs(args.output_path, exist_ok=True)
    if args.run_prefix:
        run_file = os.path.join(args.output_path,f"{args.run_prefix}_bm25_run_k1{bm25_k1}_b{bm25_b}.txt")
    else:
        run_file = os.path.join(args.output_path,f"bm25_run_k1{bm25_k1}_b{bm25_b}.txt")

    batch_size = len(query_list) if args.batch_size == -1 else args.batch_size

    with open(run_file, "w") as fw:
        for i in trange(0, len(query_list), batch_size):
            batch_query_list = query_list[i:i + batch_size]
            batch_qid_list = qid_list[i:i + batch_size]

            hits = searcher.batch_search(batch_query_list, batch_qid_list, k=topk, threads=5, fields={"contents": 1.0, "title": 1.0})

            for qid in batch_qid_list:
                for j, item in enumerate(hits[qid]):
                    fw.write("{} {} {} {} {} {}\n".format(qid, "Q0", item.docid, j + 1, item.score, "bm25"))

    return run_file


def print_res(run_file, qrel_file, rel_threshold=1,ignore_identical_ids=True):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
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
        rel =  2000 - int(line[3])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

        rank = int(line[3])
        if rank > 10:
            continue
        if query not in runs_top10:
            runs_top10[query] = {}
        runs_top10[query][passage] = rel


    if ignore_identical_ids:
        for query in list(runs.keys()):
            if query in runs[query]:
                del runs[query][query]

        for query in list(runs_top10.keys()):
            if query in runs_top10[query]:
                del runs_top10[query][query]


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
            # "MRR@10": round(np.average(mrr_list), 4),
            "NDCG@10": round(np.average(ndcg_10_list), 4),
            # "Recall@10": round(np.average(recall_10_list), 4),
            # "Recall@100": round(np.average(recall_100_list), 4),
            "Recall@1000": round(np.average(recall_1000_list), 4),
        }
    
    logger.info(res)
    return res


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_path',help='')
    parser.add_argument('--index_dir',help='')
    parser.add_argument('--output_path',help='')
    parser.add_argument('--qrel_path',help='')
    parser.add_argument('--only_eval',help='',action='store_true')
    parser.add_argument('--batch_size',help='',type=int,default=-1)
    parser.add_argument('--run_prefix',help='',default=None)
    
    args = parser.parse_args()

    if args.only_eval:
        res = print_res(run_file=args.output_path, qrel_file=args.qrel_path)
        run_file=args.output_path
    else:
        run_file = bm25(args)
        res = print_res(run_file=run_file, qrel_file=args.qrel_path, rel_threshold=1)
    with open(f"{run_file}.eval","w")as fw:
        fw.write(json.dumps(res))

if __name__=="__main__":
    main()