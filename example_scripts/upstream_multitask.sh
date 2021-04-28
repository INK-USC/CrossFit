cd ..

TASK_SPLIT=dataloader/custom_tasks_splits/random.json
CUDA_VISIBLE_DEVICES=0 \
python cli_multitask.py \
--do_train \
--train_dir data \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps 17450 \
--warmup_steps 1047 \
--model facebook/bart-base \
--output_dir models/upstream-multitask \
--train_batch_size 32 \
--num_train_epochs 10;