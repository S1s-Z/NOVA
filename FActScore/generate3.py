from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
import os
from tqdm import tqdm
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/mnt/public/shuzhengsi/factuality/models/models--MingLiiii--cherry-alpaca-5-percent-7B/snapshots/a31655f7fd37aa602764aa5f2222c1897422d347"
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)

    model.to(device)
    model.eval()


    # topics: list of strings (human entities used to generate bios)
    # generations: list of strings (model generations)

    topics = []
    generations = []
    for line in open("/home/ssz/previous_work/factuality/FActScore/data/prompt_entities.txt"): 
        topics.append(line.strip())

    outputs_list = []
    for topic in tqdm(topics):
        prompt = f"Question: Tell me a bio of " + topic + "."
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n ### Instruction:\n{prompt}\n\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        generate_ids = model.generate(input_ids, max_length=args.max_length, pad_token_id=tokenizer.eos_token_id )
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generations.append(outputs)
        output = {"topic": prompt, "output": outputs.split(prompt)[1]}
        outputs_list.append(output)
        output_dir = './my_test/' + args.model_name_or_path.replace("/","-") + '.jsonl'
        output_dir = output_dir.replace("-mnt-public-shuzhengsi-factuality-output_models-","")
        with jsonlines.open('./my_test/'+args.model_name_or_path.replace("/","-")+'.jsonl',mode='a') as writer:
            writer.write(output)
            
if __name__ == "__main__":
    main()
