import logging
import random
import sys
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel

    
def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    logger = logging.getLogger(training_args.hub_model_id)
    os.environ["WANDB_PROJECT"] = "COPO"                                                 # set wandb

    if type(data_args.dataset_mixer) == str:
        data_args.dataset_mixer = eval(data_args.dataset_mixer)
    
    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    
    if is_adapter_model(model, model_args.model_revision) is True:
        print(f"\nLoading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        print(f"\n*** Loading Base model {peft_config.base_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        print(f"\n*** Loading Base model done.")
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None
        print(f"*** Load adapter done.")

    model.eval()
    print(f"Merging with merge_and_unload...")
    base_model = model.merge_and_unload()
    
    save_dir = os.path.join(training_args.output_dir, "merge")
    print(f"saving to...", save_dir)
    tokenizer.save_pretrained(save_dir)
    base_model.save_pretrained(save_dir, safe_serialization=False)            # max_shard_size='10GB'
    logger.info("*** Merge complete! ***")

if __name__ == "__main__":
    main()
