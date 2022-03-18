#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64
export CUDA_VISIBLE_DEVICES=0


###################################################
# Working with merged country level chkpt
###################################################
tasks=(
	########################### dia2msa ########################
	ARLU_Binary_test_data_id_light.tsv
	###################LDC and IWSLT MT###############################
	# task_1.tsv
	# MT05_LDC2010T14.tsv_last.tsv
	# IWSLT16.TED.tst2010.ar-en.tsv_last.tsv 
	###################################   Transliteration ##################
	# test.tsv
	###################################   Praphrasing ##################
	# SemEval_Praph.tsv
)

models=(
    # TODO: add the model for general masked language modelling. 

	# 31M_MSA_checkpoint-2083914_msa_AR_En
	araT5_binary_msa-da-SUPER-ARLU # dialect classifier
         # TODO: 
	# "google_mT5"
)

for task in "${tasks[@]}"
do
    for model_name in "${models[@]}"
    do
        epochs_num=20
        le=5e-5 #1e-4 #
        batch_size=8 #per_devic
        ### MT
        max_source_length=256;  max_target_length=256
		text_col="content"; label_col="label"; metric_for_best_model='eval_bleu'
        
		data_dir="/project/rrg-mageed/DataBank/DIA_AR_data_bin/"$task
        out="/scratch/fsamir8/ara_dial_translation/Fixed_pred_"$task"_"$model_name"_"$le"_"$max_source_length"_"$max_target_length"_"$text_col

        mkdir -p $out
        # qsub -l walltime=04:59:59,select=1:ncpus=24:mpiprocs=1:ompthreads=24:ngpus=4:gpu_mem=32gb:mem=186gb \
		
		# qsub -A st-amuham01-1 -l walltime=00:10:59,select=1:ncpus=12:mpiprocs=1:ompthreads=12:mem=50gb \
        #     -o $log_file.log -e $log_file.err -N $task \
        #     -v label_col=$label_col,text_col=$text_col,data_dir=$data_dir,task=$task,model_name=$model_name,epochs_num=$epochs_num,le=$le,max_source_length=$max_source_length,max_target_length=$max_target_length,out=$out,batch_size=$batch_size,metric_for_best_model=$metric_for_best_model \
        #     /scratch/st-amuham01-1/elmadany/T5/run_trainier_seq2seq_huggingface_once_sockeye_moatez.sh

        # sbatch --time=02-23:59--mem=186G --gres=gpu:v100l:1 --cpus-per-task=12 --account=rrg-mageed --job-name="$model"_"$task" --output=$log_file \ run_trainier_seq2seq_huggingface_once.sh $label_col $text_col $data_dir $task $model_name $epochs_num $le $max_source_length $max_target_length $out $batch_size $metric_for_best_model    
        sbatch --export=ALL,data_dir=$data_dir,task=$task,model_name=$model_name,epochs_num=$epochs_num,le=$le,max_source_length=$max_source_length,max_target_length=$max_target_length,out=$out,batch_size=$batch_size,text_col=$text_col,label_col=$label_col,metric_for_best_model=$metric_for_best_model /project/rrg-mageed/fsamir8/arabic-dialect-machine-translation/packages/ara_t5_scripts/run_trainier_seq2seq_huggingface_once_sockeye_samir.sh 
    done
done
