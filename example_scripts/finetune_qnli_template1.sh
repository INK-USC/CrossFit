cd ..

CUDA_VISIBLE_DEVICES=1 \
python cli_singletask_template.py \
--template_id qnli_t1 \
--train_file data/glue-qnli/glue-qnli_16_100_train.tsv \
--dev_file data/glue-qnli/glue-qnli_16_100_dev.tsv \
--test_file data/glue-qnli/glue-qnli_16_100_test.tsv \
--do_train \
--do_predict \
--total_steps 1000 \
--eval_period 100 \
--warmup_steps 100 \
--model facebook/bart-base \
--output_dir models/singletask-glue-qnli-template-no-hp \
--train_batch_size 8 \
--learning_rate 1e-5;