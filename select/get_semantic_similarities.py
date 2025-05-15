import argparse
import csv
import os
import pickle
import random
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter, OrderedDict
import torch.nn.functional as F
import math
import argparse
from argparse import ArgumentParser

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


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():

    llama_version = 'llama-3-fs'
    factuality_result = load_json(f'./data/{llama_version}/consistency_result_with_response.json')
    parser = ArgumentParser()

    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=52002)
    parser.add_argument('--max_length',type=int, default=512)
    parser.add_argument('--sampling_num', type=int, default=10)
    args = parser.parse_args()
    start_index = args.start_index
    end_index = args.end_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set a seed value
    seed_value = 10
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

    for data_temp in tqdm(factuality_result[start_index:end_index]):
        # print(data_temp.keys())
        if 'response' not in data_temp.keys():
            final_score = 0.001
            data_temp['semantic_score'] = float(final_score)
            print('data without response, skip this data')
            continue
        
        generated_texts = [str(item) for item in data_temp['response'][:args.sampling_num]]
        unique_generated_texts = list(OrderedDict.fromkeys(generated_texts).keys())

        if data_temp['input'] != '':
            prompt = PROMPT_DICT_ALPACA['prompt_input'].format_map({"instruction":data_temp['instruction'], 'input':data_temp['input']})
        else:
            prompt = PROMPT_DICT_ALPACA['prompt_no_input'].format_map({"instruction":data_temp['instruction']})

        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index

        if len(unique_generated_texts) > 1:

            # Evalauate semantic similarity

            all_tested_answer_list = []
            for i, reference_answer in enumerate(unique_generated_texts):
                tested_answer_list_i = []
                for j in range(i + 1, len(unique_generated_texts)):

                    qa_1 =  unique_generated_texts[i]
                    qa_2 =  unique_generated_texts[j]
                    
                    tokenized_qa_1 = tokenizer.encode(qa_1, padding=False, return_tensors="pt", add_special_tokens=False)[0]
                    tokenized_qa_2 = tokenizer.encode(qa_2, padding=False, return_tensors="pt", add_special_tokens=False)[0]


                    if len(tokenized_qa_1) > args.max_length:
                        qa_1 = tokenizer.decode(tokenized_qa_1[-args.max_length:], skip_special_tokens=True)
                    if len(tokenized_qa_2) > args.max_length:
                        qa_2 = tokenizer.decode(tokenized_qa_2[-args.max_length:], skip_special_tokens=True)

                    input = qa_1 + ' [SEP] ' + qa_2
                    all_tested_answer_list.append(input)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    all_tested_answer_list.append(reverse_input)

            encoding = tokenizer.batch_encode_plus(all_tested_answer_list, padding=True, return_tensors='pt')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():  
                outputs = model(input_ids, attention_mask=attention_mask)
            
            prediction = outputs.logits

            predicted_label = torch.argmax(prediction, dim=1)


            i = 0
            j = i + 1

            for label_index in range(0,len(predicted_label),2):
                label = predicted_label[label_index].item()
                next_label = predicted_label[label_index+1].item()
                deberta_prediction = 1

                if 2 == label and 2 == next_label:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
                else:
                    has_semantically_different_answers = True
                    deberta_prediction = 0

                j = j + 1
                if j == len(unique_generated_texts):
                    i = i + 1
                    j = i + 1
            torch.cuda.empty_cache()


        nums_of_same_answer = {}

        for answer in generated_texts:
            if answer in nums_of_same_answer.keys():
                nums_of_same_answer[answer] += 1
            else:
                nums_of_same_answer[answer] = 1


        list_of_semantic_ids = [semantic_set_ids[x] for x in generated_texts]

        label_count = Counter(list_of_semantic_ids)  
        label_count_dict = dict(label_count)  

        test_input_sentence = data_temp['output']

        vote_dict = {}
        for index in range(len(list_of_semantic_ids)):
            qa_1 =  test_input_sentence
            qa_2 =  generated_texts[index]

            tokenized_qa_1 = tokenizer.encode(qa_1, padding=False, return_tensors="pt", add_special_tokens=False)[0]
            tokenized_qa_2 = tokenizer.encode(qa_2, padding=False, return_tensors="pt", add_special_tokens=False)[0]

            if len(tokenized_qa_1) > args.max_length:
                qa_1 = tokenizer.decode(tokenized_qa_1[-args.max_length:], skip_special_tokens=True)
            
            if len(tokenized_qa_2) > args.max_length:
                qa_2 = tokenizer.decode(tokenized_qa_2[-args.max_length:], skip_special_tokens=True)
                
            input = qa_1 + ' [SEP] ' + qa_2

            encoded_input = tokenizer.encode(input)
            prediction = model(torch.tensor(torch.tensor([encoded_input]), device=device))['logits']

            predicted_label = torch.argmax(prediction, dim=1).item()

            reverse_input = qa_2 + ' [SEP] ' + qa_1

            encoded_reverse_input = tokenizer.encode(reverse_input)
            reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device=device))['logits']

            reverse_predicted_label = torch.argmax(reverse_prediction, dim=1).item()

            torch.cuda.empty_cache()
            deberta_prediction = 1
            if 2 == predicted_label and 2 == reverse_predicted_label:
                pass
            else:
                has_semantically_different_answers = True
                deberta_prediction = 0
            
            label = list_of_semantic_ids[index]

            if deberta_prediction == 1:
                if label in vote_dict.keys():
                    vote_dict[label] += 1
                else:
                    vote_dict[label] = 0
            else:
                if label not in vote_dict.keys():
                    vote_dict[label] = 0
                else:
                    pass

        for label in vote_dict.keys():
            label_nums = list_of_semantic_ids.count(label)
            vote_dict[label] = vote_dict[label] / label_nums

        for label in label_count_dict.keys():
            label_count_dict[label] = label_count_dict[label] / len(list_of_semantic_ids)


        voted_label = max(vote_dict, key=lambda k: vote_dict[k])

        if math.isclose(vote_dict[voted_label], 0.0):
            final_score = 0.001

        else:
            final_p = label_count_dict[voted_label]
            all_prob = [final_p]

            for key in label_count_dict.keys():
                if key != voted_label:
                    all_prob.append(label_count_dict[key])
                else:
                    pass

            all_prob = torch.FloatTensor(all_prob).cpu()
            final_score = F.softmax(all_prob)[0].item()
        data_temp['semantic_score'] = float(final_score)
        # print(final_score)

        
        torch.cuda.empty_cache()

    with open(f'./data/{llama_version}/consistency_result_with_response_semantic_sampling{args.sampling_num}_{start_index}_{end_index}.json', 'w') as f:
        json.dump(factuality_result[start_index:end_index], f, indent=4)

if __name__ == '__main__':
    main()

