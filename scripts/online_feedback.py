from alignment import H4ArgumentParser, ModelArguments, DataArguments, DPOConfig
from transformers import AutoTokenizer
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from datasets import load_dataset, load_from_disk
import torch
from vllm import LLM, SamplingParams
import llm_blender
from tqdm import tqdm
import numpy as np
import os 
from filelock import FileLock

blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")
@torch.no_grad()
def generate_response_vllm(model, tokenizer, dataset):
    with torch.inference_mode():
        sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1024, stop=tokenizer.eos_token, skip_special_tokens=True)
        chosen_messages = dataset['chosen']                  # type: list
        chat_prompts = []
        for chosen_message in chosen_messages:
            prompt_message = chosen_message[:-1]             # chosen_message是一个list，前面的部分是prompt，最后部分是回复
            chat_prompts.append(tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True))  # 加一些标记，后面加上assistant方便LLM进行回复
        existing_chosen_responses = []
        for idx, r in enumerate(dataset['chosen']):
            res_content = r[1]["content"]                    # r[1]是一个dict, key是content，表示LLM的回复
            existing_chosen_responses.append(res_content)    # chosen的list
        existing_rejected_responses = []
        for idx, r in enumerate(dataset['rejected']):
            res_content = r[1]["content"]
            existing_rejected_responses.append(res_content)  # reject的list

        responses = model.generate(chat_prompts, sampling_params)                           # 批量的生成回复，需要 5min左右 -> 2000个问题
        responses_list = [response.outputs[0].text.strip() for response in responses]       # 提取回复的文字部分
        print("\nGenerate response done.", len(responses_list))

        dataset = dataset.add_column("reference_response", responses_list)                  # dataset 添加 LLM 的回复
        candidates_texts = [[responses_list[idx]] + [existing_chosen_responses[idx]] +
                            [existing_rejected_responses[idx]] for idx in range(len(responses_list))]  # list，每个元素包含三个回复：ref_model 产生的，原始数据的chosen, 原始数据的rejected
        prompts = dataset['prompt']
        # rank.shape=(2000, 3). 使用reward model进行排序，需要 10min左右 -> 2000个问题 * 3回复
        rank = blender.rank(prompts, candidates_texts, return_scores=False)             
        print("Generate rank done.", rank.shape)

        chosen_indices = np.argmin(rank, axis=1)                                            # 找到最好的回复 (2000,)
        rejected_indices = np.argmax(rank, axis=1)                                          # 找到最差的回复 (2000,)
        chosen_texts = np.array(candidates_texts)[np.arange(len(candidates_texts)), chosen_indices]             # 最好回复的list (2000,)
        rejected_texts = np.array(candidates_texts)[np.arange(len(candidates_texts)), rejected_indices]         # 最差回复的list (2000,)
        chosen_responses_dict = np.array([{"content": res, "role": "assistant"} for res in chosen_texts])       # 构造成数据集中 chosen 的格式
        rejected_responses_dict = np.array([{"content": res, "role": "assistant"} for res in rejected_texts])   # 构造成数据集中 rejected 的格式
        chosen_np = np.array(dataset['chosen'])                                             # (2000, 2) 分别存储prompt和response
        reject_np = np.array(dataset['rejected'])                                           # (2000, 2) 分别存储prompt和response
        update_chosen_column = np.column_stack((chosen_np[:, 0], chosen_responses_dict))    # (2000, 2)  将prompt和response合并起来
        update_reject_column = np.column_stack((reject_np[:, 0], rejected_responses_dict))  # (2000, 2)
        print("inference mode done.", len(chosen_messages))

    dataset = dataset.remove_columns(["chosen", "rejected", "score_chosen", "score_rejected"])
    dataset = dataset.add_column("chosen", update_chosen_column.tolist())
    dataset = dataset.add_column("rejected", update_reject_column.tolist())
    print("return dataset.")
    return dataset

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    if type(data_args.dataset_mixer) == str:
        # {'updated': 'baichenjia/COPO-Llama-3-8B-Instruct-dataset_iter_0', 'original': 'HuggingFaceH4/ultrafeedback_binarized'}
        data_args.dataset_mixer = eval(data_args.dataset_mixer)
    os.environ["WANDB_PROJECT"] = "SELM"                                                 # TODO: set wandb
    print("***** In Online Feedback:", model_args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, token="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")
    try:
        print("I am here 1")
        ref_model = LLM(model=model_args.model_name_or_path, tokenizer=model_args.model_name_or_path,
                        gpu_memory_utilization=0.6, swap_space=4, tensor_parallel_size=torch.cuda.device_count(),  # TODO: gpu_memory_utilization:0.6时需要使用双卡
                        trust_remote_code=True, dtype="auto")
        updated_dataset_name, iter_str = data_args.dataset_mixer["updated"].split("_iter")
        print("ref_model:", ref_model)
        # 处理 train 数据集，并存储在本地
        original_test_dataset = load_dataset(data_args.dataset_mixer["original"], split=data_args.dataset_splits[1])  
        new_test_dataset = generate_response_vllm(ref_model, tokenizer, original_test_dataset)    
        # TODO: 改成存储到本地.  原始: # new_test_dataset.push_to_hub(updated_dataset_name, private=False, split="test_prefs"+iter_str)
        new_test_dataset.save_to_disk(os.path.join("dataset", os.path.basename(updated_dataset_name), "test_prefs"+iter_str))
        print("save to disk down: new_test_dataset")
        
        # 用类似的流程对 train 数据集进行处理
        original_train_dataset = load_dataset(data_args.dataset_mixer["original"], split=data_args.dataset_splits[0])
        new_train_dataset = generate_response_vllm(ref_model, tokenizer, original_train_dataset)
        new_train_dataset.save_to_disk(os.path.join("dataset", os.path.basename(updated_dataset_name), "train_prefs"+iter_str))
        print("save to disk down: new_train_dataset")

        # 网速太慢
        # new_train_dataset.push_to_hub(updated_dataset_name, private=False, split="train_prefs"+iter_str)


    except Exception as e:
        print(e)
