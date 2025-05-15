import json
import argparse

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 设置命令行参数
    llama_version = 'llama-3-fs'
    sampling_num = '10'

    # 输入为已经完成排序的数据
    parser = argparse.ArgumentParser()
    parser.add_argument('--factuality_data', type=str, default=f'./data/{llama_version}/consistency_result_with_response_semantic_sampling{args.sampling_num}_0_52002.json')
    args = parser.parse_args()
    factuality_result = load_json(args.factuality_data)

    # 根据两个score进行排序
    
    for item in factuality_result:
        cons_score = item["ConScore"]
        semantic_score = item["semantic_score"]
        item["final_score"] = semantic_score / cons_score

    factuality_result = sorted(factuality_result, key=lambda x: x['final_score'], reverse=True)

    with open(f"./data/{llama_version}/sampling{sampling_num}_cons_semantic_sorted.json", 'w') as f:
        json.dump(factuality_result, f, indent=4)





if __name__ == '__main__':
    main()
