#!/bin/bash

#$ -l rt_C.small=1
#$ -l h_rt=8:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/13.2.0 cuda/11.2/11.2.2 cudnn/8.4/8.4.1 openjdk/11.0.22.0.7
conda activate pyserini

TASK=BeIR/arguana
INDEX="/home/ace14788tj/extdisk/neural_ir/bm25_searcher/beir_index/$(basename $TASK)"
INPUT="/home/ace14788tj/extdisk/neural_ir/bm25_searcher/collections/queries/$(basename $TASK)/"
TOPIC="${INPUT}$(basename $TASK).tsv"
RUN=/home/ace14788tj/extdisk/neural_ir/bm25_searcher/results/$(basename $TASK)/
QREL=/home/ace14788tj/extdisk/neural_ir/bm25_searcher/collections/qrels/$(basename $TASK)/

# python data_load.py --dataset_name ${TASK} \
#  --output_dir ${INPUT} \
#  --dataset_config queries --dataset_split queries

# python qrel_load.py --dataset_name ${TASK} --output_dir ${QREL}

python bm25_search.py --index_dir ${INDEX} \
 --query_path /home/ace14788tj/extdisk/neural_ir/gpt_rewriter/request_jsonl/arguana/arguana_request.tsv --output_path ${RUN} \
 --qrel_path "${QREL}$(basename $TASK).tsv" --run_prefix gpt_rewrite