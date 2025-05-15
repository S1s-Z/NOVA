import os
import json
from tqdm import tqdm


llama_version = 'llama-3-fs'


def merge_json_files(folder_path):
    all_data = []

    # 遍历文件夹中的文件
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    # print(len(data))
                    # print(filename)
                    # print(data)
                    for item in data:
                        del item['response']
                        all_data.append(item)
                    # # print(data[0])
                    # # del data['response']
                    # all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f'无法解析 JSON 文件 {filename}: {e}')

    # 将合并后的数据写入一个新的 JSON 文件
    print(f'Done')
    return all_data

def filter_data_without_score(full_data, computed_data):
    filter_data_without_score = []
    filter_data_with_score = []
    for full_item in tqdm(full_data):
        flag = 0
        for computed_item in computed_data:
            if full_item['output'] == computed_item['output'] and full_item['input'] == computed_item['input'] and full_item['instruction'] == computed_item['instruction']:
                flag = 1
                full_item['ConScore'] = computed_item['ConScore']

        if flag == 1:
            filter_data_with_score.append(full_item)
        if flag == 0:
            filter_data_without_score.append(full_item)
    # print(len(filter_data_without_score))
    with open(f"./data/{llama_version}/filter_data_without_score.json", 'w') as f:
        json.dump(filter_data_without_score, f, indent=4)
    return filter_data_without_score, filter_data_with_score


# 合并某个文件夹的 json 文件
folder_path = f"./data/{llama_version}/consistency_result"
computed_data = merge_json_files(folder_path)
print('Computed data:', len(computed_data))
with open(f"./data/{llama_version}/consistency_result_merged.json", 'w') as f:
    json.dump(computed_data, f, indent=4)

# 检查出没有 score 的数据
check_file_path = './data/ori/alpaca_data.json'
with open(check_file_path, 'r') as f:
    full_data = json.load(f)

print('Total:', len(full_data))
filter_data_without_score, filter_data_with_score = filter_data_without_score(full_data, computed_data)
print('Filter data without score:', len(filter_data_without_score))
print('Filter data with score:', len(filter_data_with_score))


# 找到重新计算过的，先前未计算的data
folder_new_path = f"./data/{llama_version}/consistency_result_without_score"
computed_new_data = merge_json_files(folder_new_path)
print('Computed new data:', len(computed_new_data))


#因为两次计算，采样可能导致ConScore 不一致，所以需要对数据进行去重保证数量和alpaca一致
final_consistency_data = filter_data_with_score
computed_ids = [item['id'] for item in filter_data_with_score]
no_computed_ids = [item['id'] for item in filter_data_without_score]

for item in tqdm(filter_data_without_score):
    for new_item in computed_new_data:
        if item['instruction'] == new_item['instruction'] and item['input'] == new_item['input'] and item['output'] == new_item['output']:
            item['ConScore'] = new_item['ConScore']

    if 'ConScore' in item.keys():
        final_consistency_data.append(item)
    else:
        item['ConScore'] = 0.0
        final_consistency_data.append(item)



print('Final consistency data:', len(final_consistency_data))

if len(final_consistency_data) == len(full_data):
    final_consistency_data = sorted(final_consistency_data, key=lambda x: x["ConScore"])
    with open(f"./data/{llama_version}/consistency_result.json", 'w') as f:
        json.dump(final_consistency_data, f, indent=4)
else:
    print('Error: 计算数据和原始数据长度不一致')


