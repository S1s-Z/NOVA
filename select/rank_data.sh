
# put the orinal aplaca in   ./data/ori/
# This script is very time-consuming, so if your sampling stage is interrupted, please refer to merge.py
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python multi_inner_fs.py \
--data_path ./data/ori/alpaca_data.json \
--output_dir ./data/llama-3-fs/consistency_result \
--num_process 8 \
--cuda_device 0 1 2 3 4 5 6 7 \
--batch_size 1 \
--data_split_start 0 \
--data_split_end 52002 \
--seed 0


python get_semantic_similarities_instance.py

# get the rank of familiarity.
python familiarity_rank.py

# put the ranked qulaity data in ./data/ori/
python final_rank.py