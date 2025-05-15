
import base64
import io
import multiprocessing
import random
import traceback
import argparse
from argparse import ArgumentParser
from multiprocessing import Process
from typing import Tuple, List
import numpy as np
import requests
import copy
import os, sys
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import time
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()



padding_value = 0

fs_prompt = open("fs_prompt.txt").read()

PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def get_sentence_embeddings(sampled_outputs, tokenizer, model):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.padding_side = "left"
    tokenized_sampled_outputs = tokenizer(sampled_outputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        last_hidden_state = model(**tokenized_sampled_outputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
    idx_of_the_last_non_padding_token = tokenized_sampled_outputs.attention_mask.bool().sum(1)-1
    sentence_embeddings = last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx_of_the_last_non_padding_token]

    return sentence_embeddings


def get_ConScore(sentence_embeddings, alpha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cov_matrix = torch.cov(sentence_embeddings).to(device)
    
    target_matrix = cov_matrix + alpha * torch.eye(cov_matrix.shape[0]).to(device)

    # 特征值
    evals, _ = torch.linalg.eig(target_matrix)
    # print(evals)

    return float(sum(torch.log(torch.tensor(evals))) / cov_matrix.shape[0])



@torch.no_grad()
def save_tsv(args, tokenizer, shard_id, shard, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()
    

    cnt = 0
    results = [] 
    print("=====================Strat training========================")
    for s_data in tqdm(shard):
        input_ids, attention_mask, points = s_data
        # print(point)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        generated_ids = model.generate(input_ids=input_ids, 
                                       attention_mask=attention_mask, 
                                       max_length=args.max_length, 
                                       do_sample=True, 
                                       num_return_sequences=args.sampling_num, 
                                       temperature=0.5, 
                                       top_p=0.9,
                                       top_k=5)


        # print(generated_ids.size())
        # for batch_index in range(0, args.batch_size):
        for batch_index in range(0, len(points)):
            point = copy.deepcopy(points[batch_index])
            sampled_outputs = []
            batch_outputs = tokenizer.batch_decode(generated_ids[batch_index*args.sampling_num:(batch_index+1)*args.sampling_num], skip_special_tokens=True)

            for sinlge_response in batch_outputs:
                if '### Response:' in sinlge_response:
                    # sampled_outputs.append(sinlge_response.split('### Response:')[1].split('### Instruction:')[0])
                    # sampled_outputs.append(sinlge_response.split('### Response:')[-1].split('### Instruction:')[0])
                    # sampled_outputs.append(sinlge_response.split('### Response:')[5].split('### Instruction:')[0])
                    sampled_outputs.append(sinlge_response.split('### Response:')[5].split('### Instruction:')[0].replace('Below is an instruction that describes a task. Write a response that appropriately completes the request.',''))
                    

            point['response'] = sampled_outputs
            sentence_embeddings = get_sentence_embeddings(sampled_outputs, tokenizer, model)
            score = get_ConScore(sentence_embeddings, 0.001)
            point['ConScore'] = score
            results.append(point)
            cnt += 1

        if cnt % 1000 == 0:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir,exist_ok=True)
            if cnt > 0:
                save_path = os.path.join(args.output_dir, f"cnt_{args.data_split_start}_{args.data_split_end}_{args.machine_id}_{shard_id}_{cnt // 1000}.json")
                with open(save_path, "w") as f:
                    json.dump(results, f)
                results = []
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    save_path = os.path.join(args.output_dir, f"cnt_{args.data_split_start}_{args.data_split_end}_{args.machine_id}_{shard_id}_{cnt // 1000}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"Shard {shard_id} done!")
    


class AlohaJS_Dataset(Dataset):
    def __init__(self, url_data, tokenizer, max_length):
        self.data = url_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_input, self.prompt_no_input = PROMPT_DICT_ALPACA["prompt_input"], PROMPT_DICT_ALPACA["prompt_no_input"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data_temp = self.data[idx]
        instruction = data_temp['instruction']
        if data_temp['input'] != '':
            prompt = fs_prompt + self.prompt_input.format_map({"instruction":instruction, 'input':data_temp['input']})
        else:
            prompt = fs_prompt + self.prompt_no_input.format_map({"instruction":instruction})
            
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)

        return inputs.input_ids, inputs.attention_mask, data_temp



def collate_fn(batch):  
    global padding_value

    input_ids = torch.nn.utils.rnn.pad_sequence(  
        [item[0].squeeze(0) for item in batch],
        batch_first=True, 
        padding_value=padding_value    
    )  
    attention_masks = torch.nn.utils.rnn.pad_sequence(  
        [item[1].squeeze(0) for item in batch],  
        batch_first=True,  
        padding_value=0  
    )  
    data_temps = [item[2] for item in batch]  
    return input_ids, attention_masks, data_temps  


def main():
    """Parse commandline arguments."""
    parser = ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='./data/result')

    parser.add_argument('--num_process', type=int, default=8)
    parser.add_argument('--cuda_device',  nargs='+', type=int, default=[0,1,2,3,4,5,6,7] )
    parser.add_argument('--num_machine', type=int, default=1)
    parser.add_argument('--machine_id', type=int, default=0)

    parser.add_argument("--data_path", type=str, default='./data/llama-3-fs/consistency_result.json',)
    parser.add_argument("--output_path", type=str, default='./data/llama-3-fs/consistency_result')
    parser.add_argument("--model_name_or_path", type=str, required=False, default='/mnt/public/shuzhengsi/factuality/models/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920')
    parser.add_argument('--seed', type=int, default=324)
    parser.add_argument('--data_split_start', type=int, default=0)
    parser.add_argument('--data_split_end', type=int, default=1)
    parser.add_argument("--sampling_num", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size",type=int,default=1)
    args = parser.parse_args()


    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
    #                                         trust_remote_code=True, 
    #                                         padding_side="left",
    #                                         )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                            padding_side="left",
                                            trust_remote_code=True,
                                             )
    # check the pad_token_id!!!
    global padding_value
    # padding_value = tokenizer.pad_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    padding_value = tokenizer.eos_token_id
    tokenizer.padding_side = "left" 

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


    with open(args.data_path, "r") as f:
        url_data = json.load(f)
    import copy

    url_data = url_data[args.data_split_start:args.data_split_end]
    # split into 8 machine, and pick the part of machine_id
    url_data = url_data[args.machine_id::args.num_machine]
    print(f'Processing {len(url_data)} data')
    # split url data into shards
    url_data = [url_data[i::args.num_process] for i in range(args.num_process)]

    dataloaders = [
        DataLoader(
            AlohaJS_Dataset(url_data[i], tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            prefetch_factor=4,
            collate_fn=collate_fn
        )
        for i in range(args.num_process)]

    multiprocessing.set_start_method('spawn')
    processes = []
    # cuda_device =  [int(x) for x in args.cuda_device.split(',')]
    cuda_device = args.cuda_device

    for shard_id, shard in enumerate(dataloaders):
        p = Process(
            target=save_tsv,
            args=(
                args,
                tokenizer,
                shard_id,
                shard,
                torch.device('cuda:{}'.format(cuda_device[shard_id % len(cuda_device)])),
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('Done!')

if __name__ == '__main__':
    main()
