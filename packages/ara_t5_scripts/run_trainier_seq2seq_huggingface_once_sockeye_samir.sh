#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time 02:00:00 
#SBATCH --job-name=AraDialectTranslation
#SBATCH --output=/scratch/fsamir8/ara_dial_translation/translation.out
#SBATCH --output=/scratch/fsamir8/ara_dial_translation/translation.error
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

## module load python/3.6.8
# module load py-pip/19.0.3-py3.6.8
module load gcc
module load cuda
module load openmpi
module load openblas
# TODO: do we need miniconda3?
# module load miniconda3


#######################################################
# label_col=$1
# text_col=$2
# data_dir=$3
# task=$4
# model_name=$5
# epochs_num=$6
# le=$7
# max_source_length=$8
# max_target_length=$9
# out=${10}
# batch_size=${11}
# metric_for_best_model=${12}
###########################################
train_file=$data_dir
#"/norm_dev_bib_0_bib_1_madar.tsv"
vaild_file=$data_dir
#"/norm_dev_bib_0_bib_1_madar.tsv"
test_file=$data_dir
#"/norm_dev_bib_0_bib_1_madar.tsv"
# train_file=$data_dir
# vaild_file=$data_dir
# test_file=$data_dir
# test_file="/scratch/st-amuham01-1/elmadany/T5/ArBench_data/summary/EASC/merged.tsv"
echo "task=$task"
echo "model_name="$model_name
echo "epochs_num="$epochs_num
echo "le="$le
echo "max_source_length-"$max_source_length
echo "max_target_length="$max_target_length
echo "out="$out
echo "train_file="$train_file
echo "vaild_file="$vaild_file 
echo "test_file="$test_file
echo "batch_size="$batch_size
echo "text_col="$text_col
echo "label_col="$label_col
echo "metric_for_best_model="$metric_for_best_model

mkdir -p $out/logger_out



ARA_SCRIPT_DIR="/home/fsamir8/projects/rrg-mageed/fsamir8/arabic-dialect-machine-translation/packages/ara_t5_scripts"
LOGGING_DIR="/home/fsamir8/scratch/ara_dial_translation"
MODEL_DIR="/project/rrg-mageed/DataBank/MSA_Dia_Style_Tansfer"

# source /project/rrg-mageed/fsamir8/dia_env/bin/activate
# The order of these commands seems to matter...
module load gcc/9.3.0 arrow scipy-stack
source /project/rrg-msilfver/fsamir8/py3env/bin/activate
# module load gcc/9.3.0 arrow python scipy-stack

cd "${ARA_SCRIPT_DIR}"

python "run_trainier_seq2seq_huggingface.py" \
        --logging_dir "${LOGGING_DIR}"  \
        --learning_rate $le \
        --max_target_length $max_target_length --max_source_length $max_source_length \
        --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size \
        --model_name_or_path "${MODEL_DIR}/${model_name}"   \
        --output_dir $out --overwrite_output_dir \
        --num_train_epochs $epochs_num \
        --train_file $train_file \
        --validation_file $vaild_file \
        --test_file $test_file \
        --metric_for_best_model $metric_for_best_model \
        --do_predict --predict_with_generate --task $task --text_column $text_col --summary_column $label_col

        # --do_eval if you want to evalute blue, F1, eytc
		#--do_train for fine-tunnig