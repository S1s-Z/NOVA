export OPENAI_API_KEY=
export OPENAI_API_BASE=



python generate.py --model_name_or_path your_llama_here --output_dir ./results/finish_inference/your_result.jsonl

python eval/main_eval.py \
    --input_file_path ./results/finish_inference/your_result.jsonl \
    --output_file_path ./results/finish_eval/your_result.jsonl \
    --rag_eval_type  all \
    --result_log_file_path ./results/logs/your_result.jsonl


