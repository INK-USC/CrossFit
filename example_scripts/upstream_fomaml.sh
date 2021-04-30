cd ..

TASK_SPLIT=dataloader/custom_tasks_splits/random.json
CUDA_VISIBLE_DEVICES=0 \
python cli_maml.py \
--do_train \
--learning_rate 1e-5 \
--output_dir models/upstream-fomaml \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps 6000 \
--warmup_steps 360 \
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--inner_bsz 32 \
--inner_step 4 \
--method fomaml \
--num_train_epochs 40;

