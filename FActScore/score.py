from factscore.factscorer import FactScorer
from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    fs = FactScorer(openai_key="/mnt/public/shuzhengsi/factuality/FActScore/openai_key.txt", model_name="retrieval+ChatGPT")


    # topics: list of strings (human entities used to generate bios)
    # generations: list of strings (model generations)

    topics = []
    generations = []

    with open(args.data_path, "r+") as f:
        for item in jsonlines.Reader(f):
            generations.append(item["output"])
            # topic = item["topic"].split("Question: Tell me a bio of")[1][:-1].strip()
            topic = item["topic"].split("Question: Tell me a bio of")[1].split("\n\n")[0][:-1].strip()
            topics.append(topic)
    
    out = fs.get_score(topics, generations, gamma=10, verbose=True)
    print ('FActScore: ',out["score"]) # FActScore
    print ("FActScore w/o length penalty: ", out["init_score"]) # FActScore w/o length penalty
    print ("rate of responding (not abstaining from answering): ", out["respond_ratio"]) # % of responding (not abstaining from answering)
    print ("average number of atomic facts per response: ", out["num_facts_per_response"]) # average number of atomic facts per response
    output_dict = {"Factscore": out["score"], "Factscore_without_length_penalty": out["init_score"], "rate_of_responding": out["respond_ratio"], "average_number_of_atomic_facts_per_response": out["num_facts_per_response"]}

    with open('./prediced_score/'+args.data_path.split('/')[-1].split('.')[0]+'_factscore.json', 'w', encoding='utf-8') as fw:
        json.dump(output_dict, fw, ensure_ascii=False, indent=4)
        print("Done!")
if __name__ == "__main__":
    main()
