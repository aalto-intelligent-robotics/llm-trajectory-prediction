# Exploring Large Language Models for Trajectory Prediction: A Technical Perspective

This repository contains code for the report by F. Munir, T. Mihaylova, S. Azam, T. Kucner, V. Kyrki "Exploring Large Language Models for Trajectory Prediction: A Technical Perspective", LBR at HRI 2024.


## About

This work explores use of large language models (LLMs) for trajectory prediction. 
We show that relatively small open-source models (with 7B parameters) can be fine-tuned to predict trajectory with a performance similar to much larger models.

### GPT-Driver

The work is based on [GPT-Driver](https://github.com/PointsCoder/GPT-Driver) and uses their provided dataset.

### PEFT

[PEFT](https://huggingface.co/docs/peft/index), or Parameter-Efficient Fine-Tuning (PEFT) is a HuggingFace library for adapting pre-trained models without fine-tuning the model parameters.

This repository contains the code for training and inference with [LoRA adapter](https://arxiv.org/abs/2106.09685).

## Setup

### Environment

Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

Create Conda environment:

```
conda create -n llmtp python=3.9

```

And activate it:

```
conda activate llmtp
```

Install required libraries:

```
pip install -r requirements.txt
```

### Data

We use the dataset provided by the GPT-Driver paper, and we use their training and validation split.

This folder `data` contains the training and validation data in JSON files.

The same files can be obtained by using the raw data from GPT-Driver and running the script `create_data_split.py`.

### Adapter Checkpoints

Download the saved checkpoints for LoRA adapters from the following links and unzip them:

| Model | HuggingFace model | Checkpoint |
| ----- | ----------------- | ---------- |
|Llama2-7B | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | [download](https://drive.google.com/file/d/1iZCD6sAAUi-y6gzRTwZtruiT4McbQJTj/view?usp=drive_link) |
|Llama2-7B-Chat | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [download](https://drive.google.com/file/d/1tntpTfbRR2uWXlTwYYlgC925fXkDmBNV/view?usp=drive_link) | 
| Mistral-7B | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | *tba* | 
| Zephyr-7B | [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | *tba* | 
| GPT-2 | [gpt2](https://huggingface.co/gpt2) | *tba* | 


## Running the Experiments

### Inference

The script `inference.py` needs to be executed with corresponding parameters. 
See the file `run_inference.sh` for an example.

* Pass the HF model name in the parameter `model_name`.
* Pass the path to the corresponding adapter checkpoint as a parameter `adapter_path`.

### Evaluation

The script `evaluation.py` needs to be executed with corresponsing parameters. 
See the file `run_evaluation.sh` for an example of running evaluation of all files in the output directory.

To save the evaluation results to a file, run:
```
./run_evaluation.sh > results/eval.txt
```

### Training

The script `training.py` needs to be executed with corresponsing parameters. 
See the file `run_training.sh` for an example.

* Pass the HF model name in the parameter `model_name`.

## Citation

```
@inproceedings{munir2024llmtrajpred,
    author = {Munir, Farzeen and Mihaylova, Tsvetomila and Azam, Shoaib and Kucner, Tomasz Piotr and Kyrki, Ville},
    title = {Exploring Large Language Models for Trajectory Prediction: A Technical Perspective},
    year = {2024},
    isbn = {9798400703232},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3610978.3640625},
    doi = {10.1145/3610978.3640625},
    booktitle = {Companion of the 2024 ACM/IEEE International Conference on Human-Robot Interaction},
    pages = {774–778},
    numpages = {5},
    keywords = {autonomous driving, large language models, trajectory prediction},
    location = {<conf-loc>, <city>Boulder</city>, <state>CO</state>, <country>USA</country>, </conf-loc>},
    series = {HRI '24}
}
```