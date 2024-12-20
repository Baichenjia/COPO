#!/bin/bash

#SBATCH --job-name=SELM-llama
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=48
#SBATCH --ntasks=48
#SBATCH -p optimal
#SBATCH -A optimal
#SBATCH --output=/ailab/user/baichenjia/baichenjia/SELM/logs/output-%N-%j.out    
#SBATCH --error=/ailab/user/baichenjia/baichenjia/SELM/logs/error-%N-%j.err   

set -x -e

source ~/.bashrc
# bash ~/proxy.sh
# module load curl 
# module load anaconda/2022.10
# module load cuda/12.1
source activate rlhf

export HF_ENDPOINT="https://hf-mirror.com"
# export http_proxy="https://baichenjia:UVgDVdxw6M5vbYFPv6j6yKCiYmhYZdGBDo0ftc0GvjF61REm3paOwsAt4Ftr@blsc-proxy.pjlab.org.cn:13128"
# export https_proxy="https://baichenjia:UVgDVdxw6M5vbYFPv6j6yKCiYmhYZdGBDo0ftc0GvjF61REm3paOwsAt4Ftr@blsc-proxy.pjlab.org.cn:13128"

echo "START TIME: $(date)"

export HUGGINGFACE_API_KEY="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"
export WANDB_API_KEY="a0b92158d55e0ca16cd94cbdebaef9117c99a118"
ACCELERATOR=deepspeed_zero3
export NCCL_ASYNC_ERROR_HANDLING=1

BASE_DIR=/ailab/user/baichenjia/baichenjia/SELM
cd ${BASE_DIR}

iter_num=3
for i in $(seq 1 $iter_num); do          # TODO: 这里看需要从哪儿开始
    echo "Iter $i START TIME: $(date)"
    username="baichenjia"
    name="SELM-Llama-3-8B-Instruct"
    fraction=$((61135/(iter_num)))
    training_dataset="HuggingFaceH4/ultrafeedback_binarized"
    model_name_or_path="data/${name}-iter-$((i-1))/merge"             # TODO: load from local path
    dataset_mixer="{'updated':'$username/${name}-dataset_iter_$i','original':'$training_dataset'}"
    dataset_splits=("train_prefs[$((fraction*(i-1))):$((fraction*i))]","test_prefs")
    hub_model_id="${name}-iter-$i"
    run_name="${name}-iter-$i"
    output_dir="data/$hub_model_id"
    if [ "$i" -eq 1 ]; then
        learning_rate=5e-7
        model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
    elif [ "$i" -eq 2 ]; then
        learning_rate=3e-7
    else
        learning_rate=1e-7
    fi

    # echo "** 执行 online_feedback.py **"
    # python scripts/online_feedback.py recipes/llama3-selm/selm_config_qlora.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer dataset_splits=$dataset_splits run_name="OF-iter-$i" || exit 1
    # wait 
    echo "** 执行 run_selm.py **"
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
        scripts/run_selm.py recipes/llama3-selm/selm_config_qlora.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path \
        dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir run_name=$run_name || exit 1
    wait 
    # echo "** 执行 merge_model.py **"
    # python scripts/merge_model.py recipes/llama3-selm/selm_config_qlora.yaml model_name_or_path="$output_dir/final" dataset_mixer=$dataset_mixer dataset_splits=$dataset_splits output_dir=$output_dir run_name="ME-iter-$i" || exit 1

    echo "Iter $i END TIME: $(date)"
done
