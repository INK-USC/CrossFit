cd ..

TASK_LIST=$1
CHECKPOINT=$2
IDENTIFIER=$3
GPU=$4

for TASK_NAME in $TASK_LIST
do
CUDA_VISIBLE_DEVICES=${GPU} \
python tune_hps_singletask_template.py \
--checkpoint ${CHECKPOINT} \
--template_id_list ${TASK_NAME}_t1 ${TASK_NAME}_t2 ${TASK_NAME}_t3 \
--task_dir data/glue-${TASK_NAME} \
--do_train \
--do_predict \
--learning_rate_list 1e-5 2e-5 5e-5 \
--bsz_list 2 4 8 \
--total_steps 1000 \
--eval_period 100 \
--warmup_steps 100 \
--model facebook/bart-base \
--output_dir models/${IDENTIFIER}/singletask-glue-${TASK_NAME} \
--predict_batch_size 32;
done