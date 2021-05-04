cd ..

TASK_SPLIT=dataloader/custom_tasks_splits/random.json
CUDA_VISIBLE_DEVICES=2 \
python cli_maml.py \
--do_train \
--learning_rate 1e-5 \
--output_dir models/upstream-reptile-with-dev \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps 6000 \
--warmup_steps 360 \
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--inner_bsz 16 \
--inner_step 4 \
--inner_lr 1e-5 \
--method reptile \
--num_train_epochs 40;