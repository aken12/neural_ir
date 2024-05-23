#!/bin/bash

#$ -l rt_C.large=1
#$ -l h_rt=8:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/13.2.0 cuda/11.2/11.2.2 cudnn/8.4/8.4.1 openjdk/11.0.22.0.7
conda activate pyserini

# export PYTHONPATH=/home/ace14788tj/extdisk/.pyenv/versions/anaconda3-2023.07-2/envs/pyserini:$PYTHONPATH

TASK=BeIR/hotpotqa
OUTPUT="/home/ace14788tj/extdisk/neural_ir/bm25_searcher/beir_index/$(basename $TASK)"
INPUT="/home/ace14788tj/extdisk/neural_ir/bm25_searcher/collections/corpus/$(basename $TASK)"

# python data_load.py --dataset_name ${TASK} \
#  --output_dir ${INPUT} \
#  --dataset_config corpus --dataset_split corpus

# if [ ! -f "$OUTPUT" ]; then
#     echo "Creating index..."
#     python -m pyserini.index -collection JsonCollection \
#                             -generator DefaultLuceneDocumentGenerator \
#                             -threads 64 \
#                             -input ${INPUT} \
#                             -index ${OUTPUT} \
# 							-storePositions -storeDocvectors -storeRaw
# fi



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
 --query_path ${TOPIC} --output_path ${RUN} --qrel_path "${QREL}$(basename $TASK).tsv" \
 --batch_size 1000