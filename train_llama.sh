source ~/.bashrc
bash ~/proxy.sh
module load curl 
module load anaconda/2022.10
module load cuda/12.1
source activate rlhf

export HF_ENDPOINT="https://hf-mirror.com"
export http_proxy="https://baichenjia:UVgDVdxw6M5vbYFPv6j6yKCiYmhYZdGBDo0ftc0GvjF61REm3paOwsAt4Ftr@blsc-proxy.pjlab.org.cn:13128"
export https_proxy="https://baichenjia:UVgDVdxw6M5vbYFPv6j6yKCiYmhYZdGBDo0ftc0GvjF61REm3paOwsAt4Ftr@blsc-proxy.pjlab.org.cn:13128"

echo "START TIME: $(date)"

export HUGGINGFACE_API_KEY="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"
export WANDB_API_KEY="a0b92158d55e0ca16cd94cbdebaef9117c99a118"

cd /ailab/user/baichenjia/baichenjia/SELM

iter_num=3
for i in $(seq 1 $iter_num); do
    username="baichenjia"
    name="SELM-Llama-3-8B-Instruct"
    fraction=$((61135/(iter_num)))
    training_dataset="HuggingFaceH4/ultrafeedback_binarized"
    model_name_or_path="$username/${name}-iter-$((i-1))"
    dataset_mixer="{'updated':'$username/${name}-dataset_iter_$i','original':'$training_dataset'}"
    dataset_splits=("train_prefs[$((fraction*(i-1))):$((fraction*i))]","test_prefs")
    hub_model_id="${name}-iter-$i"
    output_dir="data/$hub_model_id"
    if [ "$i" -eq 1 ]; then
        learning_rate=5e-7
        model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
    elif [ "$i" -eq 2 ]; then
        learning_rate=3e-7
    else
        learning_rate=1e-7
    fi
    python scripts/online_feedback.py recipes/llama3-selm/selm_config_full.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer dataset_splits=$dataset_splits || exit 1
    # Full Training
    # ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_selm.py recipes/llama3-selm/selm_config_full.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir || exit 1
    # Lora Training
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_selm.py recipes/llama3-selm/selm_config_qlora.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir || exit 1
done

# 以下为测试时候独立运行的脚本

dataset_mixer="{'updated':'dataset/SELM-Llama-3-8B-Instruct-dataset_iter_1','original':'HuggingFaceH4/ultrafeedback_binarized'}"
hub_model_id="SELM-Llama-3-8B-Instruct-iter-1"
output_dir="data/$hub_model_id"

# # Q-Lora
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_selm.py recipes/llama3-selm/selm_config_qlora.yaml model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_selm.py recipes/llama3-selm/selm_config_qlora.yaml model_name_or_path="data/Meta-Llama-3-8B-Instruct-iter-1/final" dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir

# # Full Training
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_selm.py recipes/llama3-selm/selm_config_full.yaml model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir
