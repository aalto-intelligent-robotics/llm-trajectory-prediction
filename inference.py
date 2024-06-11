import re
import argparse
import os
import torch
import numpy as np
import textwrap
import ndjson
import bitsandbytes as bnb
import tqdm
import ast
from functools import partial
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
from transformers import pipeline, Conversation
from transformers.pipelines.pt_utils import KeyDataset


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{40960}MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """

    END_KEY = "### End"

    message = sample["messages"]
    system = f"{message[0]['role']}\n{message[0]['content']}"
    user = f"{message[1]['role']}\n{message[1]['content']}"
    assistant = f"{message[2]['role']}\n{message[2]['content']}"
    end = f"{END_KEY}"

    parts = [part for part in [system, user, assistant, end] if part]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt

    return sample


def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in [
        "n_positions",
        "max_position_embeddings",
        "seq_length",
    ]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer: AutoTokenizer, max_length: int):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def extract_first_segment(text):
    # Define the pattern to match the first segment before "### Endpoint"
    pattern = re.compile(r"(.*?)(?=\n### |$)", re.DOTALL)

    # Use the pattern to find the match in the given text
    match = re.search(pattern, text)

    # Extract the matched segment
    if match:
        return match.group(1).strip()
    else:
        return None


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(
    tokenizer: AutoTokenizer, max_length: int, seed: int, dataset: str
):
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    )

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: dict,
    output_path: str,
    device: str,
):
    """Evaluate the model on the given data.

    Args:
        model: PerfModel
        tokenizer: AutoTokenizer
        dataset: Validation dataset
        output_path: Path to where to save the JSON file containing predicted messages
        device: cuda or cpu
    """

    pipe = pipeline("conversational", model=model, tokenizer=tokenizer)

    for message in dataset:
        con = [message["messages"][0], message["messages"][1]]
        asst = pipe(con)

        result = extract_first_segment(asst.generated_responses[-1])
        output_dict = {
            "GPT": result,
            "GT": message["messages"][2]["content"],
            "token": message["token"],
        }
        drt = [output_dict]
        with open(output_path, "a+", encoding="utf-8") as file:
            file.write(ndjson.dumps(drt) + "\n")


def main():
    """
    Main function.
    - reads saved checkpoint
    - runs inference on a given validation set
    - writes the result to a file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_data_file",
        dest="validation_data_file",
        type=str,
        default=None,
        required=True,
        help="File with validation data in SON format. ",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        required=True,
        help="Model name.",
    )
    parser.add_argument(
        "--adapter_path",
        dest="adapter_path",
        type=str,
        default=None,
        required=True,
        help="Path to the saved adapter checkpoint.",
    )
    parser.add_argument(
        "--results_file",
        dest="results_file",
        type=str,
        default=None,
        required=True,
        help="Path to the file in which results will be written.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        required=False,
        help="Use cpu or cuda.",
    )
    parser.add_argument(
        "--cache_dir",
        dest="cache_dir",
        type=str,
        default=None,
        required=False,
        help="The cache directory to save the downloaded models.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        help="Random seed.",
    )
    args = parser.parse_args()
    model_name = args.model_name
    validation_data_file = args.validation_data_file
    adapter_path = args.adapter_path
    results_file = args.results_file
    device = args.device
    seed = args.seed
    cache_dir = args.cache_dir
    hf_token = os.getenv("HF_ACCESS_TOKEN")

    dataset = load_dataset(
        "json", data_files=validation_data_file, split="train"
    )
    print(f"Number of prompts: {len(dataset)}")
    print(f"Column names are: {dataset.column_names}")

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, cache_dir=cache_dir
    )

    chat = [
        {
            "role": "system",
            "content": "**Autonomous Driving Planner**  Role: You are the brain of an autonomous vehicle.  Plan a safe 3-second driving trajectory. Avoid collisions with other objects. Context- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction youre facing. Youre at point (0,0).- Objective: Create a 3-second route using 6 waypoints, one every 0.5 seconds.Inputs 1. Perception & Prediction: Info about surrounding objects and their predicted movements. 2. Historical Trajectory: Your past 2-second route, given by 4 waypoints. 3. Ego-States: Your current state including velocity, heading angular velocity, can bus data, heading speed, and steering signal. 4. Mission Goal: Goal location for the next 3 seconds. Task - Thought Process: Note down critical objects and potential effects from your perceptions and predictions.Action Plan: Detail your meta-actions based on your analysis.Trajectory Planning: Develop a safe and feasible 3-second route using 6 new waypoints.Output- Thoughts:  - Notable Objects    Potential Effects- Meta Action- Trajectory (MOST IMPORTANT):  - [(x1,y1), (x2,y2), ... , (x6,y6)]",
        },
        {
            "role": "user",
            "content": "Perception and Prediction: - trailer at (-18.00,11.69), moving to (-2.31,16.57). - trafficcone at (3.51,1.45), moving to (3.53,1.47). - adult at (4.91,3.36), moving to (5.08,2.40). - truck at (-9.90,13.49), moving to (5.72,18.62). - adult at (10.52,15.46), moving to (10.62,15.23). - adult at (5.65,3.59), moving to (5.64,1.51).Ego-States: - Velocity (vx,vy): (-0.00,0.00) - Heading Angular Velocity (v_yaw): (-0.00) - Acceleration (ax,ay): (0.00,0.00) - Can Bus: (-0.11,0.08) - Heading Speed: (0.00) - Steering: (0.14) Historical Trajectory (last 2 seconds): [(0.00,-0.00), (0.00,-0.00), (0.00,-0.00), (0.00,-0.00)] Mission Goal: FORWARD",
        },
        {
            "role": "assistant",
            "content": "Thoughts: - Notable Objects from Perception: None  Potential Effects from Prediction: None Meta Action: STOP Trajectory:[(-0.00,0.00), (-0.00,0.00), (-0.00,0.00), (-0.00,0.00), (-0.00,0.00), (-0.00,0.00)]",
        },
    ]
    tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    max_length = get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
    print(f"Column names are: {dataset.column_names}")

    inference(model, tokenizer, dataset, results_file, device=device)


if __name__ == "__main__":
    main()
