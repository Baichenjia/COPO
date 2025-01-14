source ~/.bashrc
source activate rlhf

export HF_ENDPOINT="https://hf-mirror.com"

echo "START TIME: $(date)"

export HUGGINGFACE_API_KEY="xxxxxxxx"     # TODO: your huggingface api key
export WANDB_API_KEY="xxxxxxxx"           # TODO: your wandb api key
ACCELERATOR=deepspeed_zero3
export NCCL_ASYNC_ERROR_HANDLING=1

BASE_DIR=/Path-to-COPO/COPO/              # TODO: your dir to COPO

cd ${BASE_DIR}

iter_num=3                                # run alignment for 3 iterations
for i in $(seq 1 $iter_num); do
    echo "Iter $i START TIME: $(date)"
    username="baichenjia"
    name="COPO-Llama-3-8B-Instruct"
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

    echo "** RUN online_feedback.py **"
    python scripts/online_feedback.py recipes/llama3-copo/copo_config_qlora.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer dataset_splits=$dataset_splits run_name="OF-iter-$i" || exit 1
    wait 

    echo "** RUN count_trainer.py **"
    accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 \
        scripts/count_trainer.py recipes/llama3-copo/copo_config_qlora.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer \
        hub_model_id=$hub_model_id output_dir="$output_dir/vhead" run_name="Count-iter-$i" per_device_train_batch_size=1 per_device_eval_batch_size=1 learning_rate=1e-4
    wait 
    
    echo "** RUN run_copo_count.py **"
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
        scripts/run_copo.py recipes/llama3-copo/copo_config_qlora.yaml learning_rate=$learning_rate model_name_or_path=$model_name_or_path \
        dataset_mixer=$dataset_mixer hub_model_id=$hub_model_id output_dir=$output_dir run_name=$run_name || exit 1    
    wait     
    
    echo "** RUN merge_model.py **"
    python scripts/merge_model.py recipes/llama3-copo/copo_config_qlora.yaml model_name_or_path="$output_dir/final" dataset_mixer=$dataset_mixer dataset_splits=$dataset_splits output_dir=$output_dir run_name="ME-iter-$i" || exit 1

    echo "Iter $i END TIME: $(date)"
done
