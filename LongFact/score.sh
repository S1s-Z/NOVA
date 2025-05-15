





type=objects
python generate.py --type ${type} --model_name_or_path your_llama_here --output_dir ./data_inferenced_${type}/your_json_file.json
python divide_atomic_facts.py --eval_dir data_inferenced_${type}/ --file_name your_json_file.json
python evaluate.py --eval_dir data_inferenced_${type}/atomic_facts --file_name your_json_file.json


type=concepts
python generate.py --type ${type} --model_name_or_path your_llama_here --output_dir ./data_inferenced_${type}/your_json_file.json
python divide_atomic_facts.py --eval_dir data_inferenced_${type}/ --file_name your_json_file.json
python evaluate.py --eval_dir data_inferenced_${type}/atomic_facts --file_name your_json_file.json