NAME=$1
POSTFIX=$2
python collect_results.py \
--logs_dir ../models/${NAME} \
--output_file ../playground/results_summary/${NAME}.csv \
--postfix ${POSTFIX};
