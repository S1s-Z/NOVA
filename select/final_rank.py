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
    parser.add_argument('--quality_data', type=str, default='./data/ori/car_quality_result.json')
    parser.add_argument('--factuality_data', type=str, default=f'./data/{llama_version}/sampling{sampling_num}_cons_semantic_sorted.json')
    args = parser.parse_args()
    quality_result = load_json(args.quality_data)
    factuality_result = load_json(args.factuality_data)


    # 根据两个rank进行排序
    index = 1
    for quality_item in quality_result:
        quality_item['index'] = index
        index += 1
    
    index = 1
    for factuality_item in factuality_result:
        factuality_item['index'] = index
        index += 1

    for i in range(len(factuality_result)):
        if type(factuality_result[i]['output']) == dict:
            # print(factuality_result[i]['output'])
            factuality_result[i]['output'] = str(factuality_result[i]['output'])
            pass
        if type(quality_result[i]['output']) == dict:
            quality_result[i]['output'] = str(quality_result[i]['output'])


    for i in range(len(factuality_result)):
        factuality_result[i]['order_index'] = factuality_result[i]['index'] + quality_result[i]['index']


    factuality_result = sorted(factuality_result, key=lambda x: x['order_index'])

    #  综合考虑quality 、se 、 cons
    # 注意数据比例
    # gpt 003
    with open(f"./data/{llama_version}/sampling{sampling_num}_se_cons_quality_5percent.json", 'w') as f:
        json.dump(factuality_result[0:5000], f, indent=4)





if __name__ == '__main__':
    main()
