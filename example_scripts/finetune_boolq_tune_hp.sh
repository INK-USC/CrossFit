cd ..

python tune_hps_singletask.py \
--task_dir data/boolq/ \
--do_train \
--do_predict \
--learning_rate_list 1e-5 2e-5 5e-5 \
--bsz_list 2 4 8 \
--total_steps 1000 \
--eval_period 100 \
--warmup_steps 100 \
--model facebook/bart-base \
--output_dir models/singletask-boolq \
--predict_batch_size 32 \