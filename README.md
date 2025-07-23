# NOVA


The code of our paper "Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering".

## 🎇Overview

Training LLMs on data that contains unfamiliar knowledge during the instruction tuning stage can make LLMs overconfident and encourage hallucinations. To address this challenge, we introduce a novel framework, **NOVA**, which identifies high-quality data that aligns well with the LLM's learned knowledge to reduce hallucinations. **NOVA** includes Internal Consistency Probing (ICP) and Semantic Equivalence Identification (SEI) to measure how familiar the LLM is with instruction data. Specifically, ICP evaluates the LLM's understanding of the given instruction by calculating the tailored consistency among multiple self-generated responses. SEI further assesses the familiarity of the LLM with the target response by comparing it to the generated responses, using the proposed semantic clustering and well-designed voting strategy. Finally, we introduce an expert-aligned reward model, considering characteristics beyond just familiarity to enhance data quality. By considering data quality and avoiding unfamiliar data, we can utilize the selected data to effectively align LLMs to follow instructions and hallucinate less. Extensive experiments and analysis show that **NOVA** significantly reduces hallucinations and allows LLMs to maintain a strong ability to follow instructions.

## 🎯 Usage

### 🔎 Setup

Install the environments with pip: `pip install -r requirements.txt`. 

Meanwhile, our training code is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Please correctly install the environments and set up the dataset dir according to LLaMA-Factory.

### 📢 Select

You can find the corresponding code in `/select`. 

We provide the code to calculate the designed familiarity. For training expert-aligned reward model for ensuing the data quality, please kindly refer to [CaR Repo/Ranking](https://github.com/IronBeliever/CaR).

```sh
sh rank_data.sh
```


### 📢 Train

You can find the corresponding script in `train_models.sh`.

You can download and save the processed data through the [Tsinghua Drive/NOVA_datasets/](https://cloud.tsinghua.edu.cn/d/1f0a434da3314c8a9912/) to train the model. Please correctly put the data according to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```sh
sh train_models.sh
```

## 🎲 Evaluation

You can download and save the processed data through the [Tsinghua Drive/NOVA_datasets](https://cloud.tsinghua.edu.cn/d/1f0a434da3314c8a9912/). Plz put the correct dataset files in the correct path, e.g., `LongFact/dataset`.

### 🔍 FActScore

You can find the corresponding code in `/FActScore`.  Meanwhile, you may need to download the database used to retrieve facts according to [FActScore Repo](https://github.com/shmsw25/FActScore).

```sh
sh score.sh
```

### 🔍 FollowRAG

You can find the corresponding code in `/FollowRAG`. 

```sh
sh score.sh
```

### 🔍 LongFact

You can find the corresponding code in `/LongFact`. 

```sh
sh score.sh
```

### 🔍 MT-Bench

To reproduce our results on other benchmarks, we refer to the code in [FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) for evaluate MT-Bench tasks. 

## 🤖 All available models

Here is the full list of models we released:

| Model                                    | Checkpoint                                                         | Description                                                  |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **NOVA-LLaMA-3-8B-Alpaca-5percent**      | [🤗 Link](https://huggingface.co/ssz1111/NOVA-LLaMA-3-8B-Alpaca-5percent) | Chat model, based on LLaMA3-Base-8B, trained on selected 5% data from Alpaca. |
| **NOVA-LLaMA-3-8B-Alpaca-10percent**     | [🤗 Link](https://huggingface.co/ssz1111/NOVA-LLaMA-3-8B-Alpaca-10percent) | Chat model, based on LLaMA3-Base-8B, trained on selected 10% data from Alpaca. |
| **NOVA-LLaMA-3-8B-Alpaca-15percent**     | [🤗 Link](https://huggingface.co/ssz1111/NOVA-LLaMA-3-8B-Alpaca-15percent) | Chat model, based on LLaMA3-Base-8B, trained on selected 15% data from Alpaca. |
| **NOVA-LLaMA-3-8B-AlpacaGPT4-5percent**  | [🤗 Link](https://huggingface.co/ssz1111/NOVA-LLaMA-3-8B-AlpacaGPT4-5percent) | Chat model, based on LLaMA3-Base-8B, trained on selected 5% data from AlpacaGPT4. |
| **NOVA-LLaMA-3-8B-AlpacaGPT4-10percent** | [🤗 Link](https://huggingface.co/ssz1111/NOVA-LLaMA-3-8B-AlpacaGPT4-10percent) | Chat model, based on LLaMA3-Base-8B, trained on selected 10% data from AlpacaGPT4. |
| **NOVA-LLaMA-3-8B-AlpacaGPT4-15percent** | [🤗 Link](https://huggingface.co/ssz1111/NOVA-LLaMA-3-8B-AlpacaGPT4-15percent) | Chat model, based on LLaMA3-Base-8B, trained on selected 15% data from AlpacaGPT4. |

## ✍🏻 Citation

@inproceedings{si-etal-2025-aligning,
    title = "Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering",
    author = "Si, Shuzheng and Zhao, Haozhe and Chen, Gang and Gao, Cheng and Bai, Yuzhuo and Wang, Zhitong and An, Kaikai and Luo, Kangyang and Qian, Chen and Qi, Fanchao and Chang, Baobao and Sun, Maosong",
    editor = "Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.804/",
    pages = "16469--16488",
    ISBN = "979-8-89176-251-0",
    abstract = "Training LLMs on data containing unfamiliar knowledge during the instruction tuning stage can encourage hallucinations. To address this challenge, we introduce NOVA, a novel framework designed to identify high-quality data that aligns well with the LLM{'}s learned knowledge to reduce hallucinations. NOVA includes Internal Consistency Probing (ICP) and Semantic Equivalence Identification (SEI) to measure how familiar the LLM is with instruction data. Specifically, ICP evaluates the LLM{'}s understanding of the given instruction by calculating the tailored consistency among multiple self-generated responses. SEI further assesses the familiarity of the LLM with the target response by comparing it to the generated responses, using the proposed semantic clustering and well-designed voting strategy. Finally, to ensure the quality of selected samples, we introduce an expert-aligned reward model, considering characteristics beyond just familiarity. By considering data quality and avoiding unfamiliar data, we can utilize the selected data to effectively align LLMs to follow instructions and hallucinate less. Experiments show that NOVA significantly reduces hallucinations while maintaining a competitive ability to follow instructions."
}











