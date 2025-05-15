from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
import os
from tqdm import tqdm
import jsonlines


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/mnt/public/shuzhengsi/factuality/models/models--MingLiiii--cherry-alpaca-5-percent-7B/snapshots/a31655f7fd37aa602764aa5f2222c1897422d347"
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default='./results/finish_inference/data_inferenced.jsonl')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    followRAG_full = load_json('./followRAG/followRAG_full.json')

    model.to(device)
    model.eval()


    # topics: list of strings (human entities used to generate bios)
    # generations: list of strings (model generations)

    outputs_list = []


    # generation = []

    for dp in tqdm(followRAG_full):

        prompt = dp['prompt']
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]

        if len(tokenized_prompt) > args.max_length:
            half = int(args.max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)


        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        generate_ids = model.generate(input_ids, max_length=args.max_length, pad_token_id=tokenizer.eos_token_id)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        dp['response'] = outputs
        # generation.append(dp)

        with jsonlines.open(args.output_dir, mode='a') as writer:
            writer.write(dp)
            
if __name__ == "__main__":
    main()
