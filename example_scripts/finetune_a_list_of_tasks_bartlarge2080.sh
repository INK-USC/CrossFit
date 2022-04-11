cd ..

TASKS=$1
CHECKPOINT=$2
IDENTIFIER=$3
GPU=$4


for TASK in $TASKS
do

echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=$GPU \
python tune_hps_singletask.py \
--task_dir data/${TASK}/ \
--checkpoint $CHECKPOINT \
--memory_efficient \
--do_train \
--do_predict \
--learning_rate_list 1e-5 2e-5 5e-5 \
--bsz_list 8 4 2 \
--total_steps 1000 \
--eval_period 100 \
--warmup_steps 100 \
--max_grad_norm 0.1 \
--weight_decay 0.01 \
--model facebook/bart-large \
--output_dir models/${IDENTIFIER}/singletask-${TASK} \
--gradient_accumulation_steps 1 \
--num_train_epochs 1000;

done