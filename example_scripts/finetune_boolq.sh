cd ..

CUDA_VISIBLE_DEVICES=2 \
python cli_singletask.py \
--train_file data/boolq/boolq_16_100_train.tsv \
--dev_file data/boolq/boolq_16_100_dev.tsv \
--test_file data/boolq/boolq_16_100_test.tsv \
--do_train \
--do_predict \
--total_steps 1000 \
--eval_period 100 \
--warmup_steps 100 \
--model facebook/bart-base \
--output_dir models/singletask-boolq-no-hp \
--train_batch_size 2 \
--learning_rate 1e-5;