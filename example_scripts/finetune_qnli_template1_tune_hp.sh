cd ..

CUDA_VISIBLE_DEVICES=3 \
python tune_hps_singletask_template.py \
--template_id qnli_t1 \
--task_dir data/glue-qnli \
--do_train \
--do_predict \
--learning_rate_list 1e-5 2e-5 5e-5 \
--bsz_list 2 4 8 \
--total_steps 1000 \
--eval_period 100 \
--warmup_steps 100 \
--model facebook/bart-base \
--output_dir models/singletask-glue-qnli-template \
--predict_batch_size 32;