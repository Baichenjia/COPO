from trl import AutoModelForCausalLMWithValueHead
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
from accelerate import PartialState
import numpy as np
import torch
import copy
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from dataclasses import dataclass
import transformers
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl.trainer.utils import pad_to_length
from trl.import_utils import is_peft_available
from transformers import AutoModelForCausalLM
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    pad_to_length
)
from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_quantization_config,
    get_tokenizer,
)
from peft import PeftConfig, PeftModel
import os, sys
os.environ['CURL_CA_BUNDLE'] = ''


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


class CoinFlipMaker(object):
    def __init__(self, output_dimensions=20, p_replace=1.0, only_zero_flips=False):
        self.p_replace = p_replace
        self.output_dimensions = output_dimensions
        self.only_zero_flips = only_zero_flips
        self.previous_output = self._draw()

    def _draw(self, batch_size=100):
        # print("In CoinFlipMaker, _draw:", batch_size)
        if self.only_zero_flips:
            return np.zeros(self.output_dimensions, dtype=np.float32)
        sample = 2 * np.random.binomial(1, 0.5, size=(batch_size, self.output_dimensions)) - 1
        # print("In CoinFlipMaker, sample:", sample.shape)
        return sample

    def __call__(self, batch_size):
        if self.only_zero_flips:
            return np.zeros(self.output_dimensions, dtype=np.float32)
        new_output = self._draw(batch_size)
        self.previous_output = new_output
        return new_output

    def reset(self):
        if self.only_zero_flips:
            self.previous_output = np.zeros(self.output_dimensions, dtype=np.float32)
        self.previous_output = self._draw()



class CountValueHead(nn.Module):     # COPO: the value head of CFN
    def __init__(self, **kwargs):    # config 
        super().__init__()
        summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        # print("In CountValueHead, hidden_size:", hidden_size)
        self.hidden = nn.Linear(4096, 32) 
        self.summary = nn.Linear(32, 20)
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten()

    def forward(self, hidden_states, return_last=True):
        output = self.dropout(hidden_states)

        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.hidden(output)
        output = self.lrelu(output)
        output = self.summary(output)

        if return_last:
            output = output[..., -1, :]
        # print("CountValueHead output:", output.shape)  # (8, 20)
        return output


class AutoModelForCausalLMWithCountValueHead(AutoModelForCausalLMWithValueHead):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        """
        Initializes the model.
        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model, **kwargs)
        for name, module in pretrained_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
        print(f"Pretrained Model Detach.")

        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.value_head = CountValueHead(**v_head_kwargs)     # COPO: add the value head
        self._init_weights(**v_head_kwargs)
        
    def forward(
        self,
        concatenated_batch,
        **model_kwargs,
    ):
        # Applies a forward pass to the wrapped model and returns the logits of the value head.
        last_hidden_state = self.pretrained_model(
            concatenated_batch["reference_input_ids"],
            attention_mask=concatenated_batch["reference_attention_mask"],
            output_hidden_states=True,     # TODO
            **model_kwargs,
        ).hidden_states[-1]
        if last_hidden_state.device != self.value_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.value_head.summary.weight.device)
        value_output = self.value_head(last_hidden_state.detach())                              
        
        return value_output
    

loss_fn = torch.nn.MSELoss(reduction='mean')
coin_flip_maker = CoinFlipMaker()


class CountTrainer(Trainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        
        print("model_init_kwargs:", model_init_kwargs)
        model = AutoModelForCausalLMWithCountValueHead(model, **model_init_kwargs)
        
        self.is_encoder_decoder = model.config.is_encoder_decoder   # False
        
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self._peft_has_been_casted_to_bf16 = False

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.dataset_num_proc = dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError("You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`.")
        
    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        batch = {}
        prompt = feature["prompt"]
        # chosen = feature["chosen"]
        # rejected = feature["rejected"]
        if "reference_response" not in feature:
            return super().tokenize_row(feature)
        else:
            reference_response = feature["reference_response"]

            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(reference_response, str):
                raise ValueError(f"reference response should be an str but got {type(reference_response)}")
            reference_tokens = self.build_tokenized_answer(prompt, reference_response)

            reference_prompt_len_input_ids = len(reference_tokens["prompt_input_ids"])
            prompt_len_input_ids = reference_prompt_len_input_ids

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
            reference_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + reference_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            reference_tokens["prompt_attention_mask"] = [1] + reference_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            reference_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            reference_tokens["attention_mask"].append(1)

            longer_response_length = len(reference_tokens["input_ids"])
            # if combined sequence is too long, truncate the prompt for answer_tokens in [chosen_tokens, rejected_tokens, reference_tokens, prompt_tokens]:
            for answer_tokens in [reference_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [reference_tokens]:  
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            reference_sequence_tokens = {
                k: reference_tokens[f"prompt_{k}"] + reference_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            reference_sequence_tokens["labels"] = reference_sequence_tokens["input_ids"][:]
            reference_sequence_tokens["labels"][: len(reference_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(reference_tokens["prompt_input_ids"])

            for k, toks in {
                "reference_": reference_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens
            return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}
        max_length = batch['reference_input_ids'].shape[1]   # TODO: max_length 的设置
        
        for k in batch:
            if k.startswith("reference") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_batch[k] = pad_to_length(batch[k], max_length, pad_value=pad_value).to(device=device)

        return concatenated_batch

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> torch.FloatTensor:
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )

        model_kwargs = ({
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),}
            if self.is_encoder_decoder
            else {})
        # TODO: 将基础模型的输出作为 value head 的输入
        value_output = model(concatenated_batch, **model_kwargs)
        # print("in concatenated_forward: value_output:", value_output.shape)  #  torch.Size([8, 20])
        return value_output
       
    def optimistic_count_loss(self, value_output: torch.FloatTensor) -> torch.FloatTensor:
        # COPO: use the coin-flip target to update the CFN network
        coin_flip_target = coin_flip_maker(batch_size=value_output.shape[0])
        if self.is_deepspeed_enabled:
            coin_flip_target = torch.from_numpy(coin_flip_target).to(dtype=torch.bfloat16, device=self.accelerator.device)
        else:
            coin_flip_target = torch.from_numpy(coin_flip_target).to(dtype=torch.float32, device=self.accelerator.device)

        assert value_output.shape == coin_flip_target.shape 
        losses = loss_fn(value_output.to(self.accelerator.device), coin_flip_target)       
        return losses
       
    def compute_loss(self, model: Union[PreTrainedModel, nn.Module], inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.FloatTensor:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator")
        
        value_output = self.concatenated_forward(model, inputs)
        loss = self.optimistic_count_loss(value_output)
        
        loss = loss.to(self.args.device)
        # print("value loss:", loss)
        return loss

    def load_value_model(self, save_dir):
        model_path = os.path.join(save_dir, "vhead.pth")
        print(f"Loading value head checkpoint to {model_path}")
        self.model.value_head.load_state_dict(torch.load(model_path))

    def save_value_model(self, save_dir):
        last_hidden_state = torch.rand(8, 539, 4096)
        if last_hidden_state.device != self.model.value_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.model.value_head.summary.weight.device)
        value_output = self.model.value_head(last_hidden_state)
        print("In accelerator_save_model:", value_output.shape)

        print("\n Direct Save:")
        torch.save(self.model.value_head, os.path.join(save_dir, "vhead.pickle"))
        
        print("\n Save State Dict:")
        state_dict = self.model.value_head.state_dict()
        for var_name in state_dict:
            print(var_name, "\t", state_dict[var_name])
        torch.save(state_dict, os.path.join(save_dir, "vhead.pth"))

        
if __name__ == "__main__":    
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    logger = logging.getLogger(training_args.hub_model_id)
    os.environ["WANDB_PROJECT"] = "COPO"                           # TODO: set wandb

    if type(data_args.dataset_mixer) == str:
        data_args.dataset_mixer = eval(data_args.dataset_mixer)
    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits, task="copo")
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "copo",           # TODO
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected", "text_response": "reference_response"}
        )
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype))
    
    print("\n model_args:", model_args, "\n")
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

    # Instantiate Counter Trainer
    logger.info(f"*** Start CounterTrainer...")
    trainer = CountTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=raw_datasets["train"],
        # train_dataset=raw_datasets["test"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
    )

    # Training loop
    train_result = trainer.train()
    
    # save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training complete ***")

    # save model
    logger.info("*** Save model ***", trainer.args.output_dir)
    trainer.save_value_model(trainer.args.output_dir)
        
    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    logger.info("*** Counter training complete! ***")
