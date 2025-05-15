from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
import os
from tqdm import tqdm
import jsonlines
from datasets import load_dataset


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

cur_dir = os.path.dirname(__file__)

def load_longfact(type="objects", max_sample_num=120):
    data_file = os.path.join(cur_dir, f"dataset/longfact_{type}_random.json")
    dataset = json.load(open(data_file, "r"))
    return dataset[:max_sample_num]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/mnt/user/wangzhitong/ssz/factuality/models/models--meta-llama--Llama-3.1-8B/snapshots/8d10549bcf802355f2d6203a33ed27e81b15b9e5"
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default='./data_inferenced/data_inferenced.jsonl')
    parser.add_argument("--type", type=str, default="objects")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    model.to(device)
    model.eval()


    data = load_longfact(type=args.type)
    # + load_longfact(type="concepts")
    # data = load_longfact(type="objects") + load_longfact(type="concepts")

    questions = [sample["prompt"] for sample in data]
    outputs_list = []

    i = 0
    for dp in tqdm(questions):
        
        instruction = dp
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n ### Instruction:\n{instruction}\n\nn### Response:" 
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]

        if len(tokenized_prompt) > args.max_length:
            half = int(args.max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)


        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        generate_ids = model.generate(input_ids, max_length=args.max_length, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        outputs_list.append({
            "question": prompt,
            "generated_answer": response.split(prompt)[1].strip(),
            "question_index": i
        })
        i = i + 1

    with open(args.output_dir, "w") as f:
        json.dump(outputs_list, f)

if __name__ == "__main__":
    main()
