import os
import numpy as np
import numba
import pickle
from tqdm import tqdm
import json
from collections import defaultdict

from neural_ir.splade_searcher.indexing.array import IndexDictOfArray
from neural_ir.utils import tensor_to_list

import torch


class SparseRetrieval:
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, index_dir_path, retrieval_output_path, dim_voc, top_k):
        self.sparse_index = IndexDictOfArray(index_dir_path, dim_voc=dim_voc)
        self.doc_ids = pickle.load(open(os.path.join(index_dir_path, "doc_ids.pkl"), "rb"))
        self.top_k = top_k
        self.retrieval_output_path = retrieval_output_path

        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        
    
    def retrieve(self, qid2emb):
        res = defaultdict(dict)
        for qid in tqdm(qid2emb):
            query_emb = qid2emb[qid]
            query_emb = query_emb.view(1, -1)
            row, col = torch.nonzero(query_emb, as_tuple=True)
            values = query_emb[tensor_to_list(row), tensor_to_list(col)]
            threshold = 0
            filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                self.numba_index_doc_values,
                                                                col.cpu().numpy(),
                                                                values.cpu().numpy().astype(np.float32),
                                                                threshold=threshold,
                                                                size_collection=self.sparse_index.nb_docs())
            # threshold set to 0 by default, could be better
            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=self.top_k)
            for id_, sc in zip(filtered_indexes, scores):
                res[str(qid)][str(self.doc_ids[id_])] = float(sc)
            
        with open(os.path.join(self.retrieval_output_path, "run.txt"), "w") as f:
            for qid in res:
                for rank,did in enumerate(res[qid]):
                    f.write(os.path.join([qid,"Q0",did,rank+1,res[qid][did],"splade"]))
                    f.write('\n')
                    
        print("Write the retrieval result to {} successfully.".format(self.retrieval_output_path))

        return res