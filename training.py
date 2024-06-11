import os
import argparse
from functools import partial
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
import torch
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
from trl import SFTTrainer
from datasets import load_dataset, Dataset


def load_model(model_name, bnb_config, cache_dir, security_token):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{40960}MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available resources
        max_memory={i: max_memory for i in range(n_gpus)},
        cache_dir=cache_dir,
        token=security_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, token=security_token
    )
    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_prompt_formats(sample, for_validation: bool = False):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    message = sample["messages"]
    system = f"{message[0]['role']}\n{message[0]['content']}"
    user = f"{message[1]['role']}\n{message[1]['content']}"
    assistant = f"{message[2]['role']}\n{message[2]['content']}"
    end = "### End"

    if for_validation:
        parts = [part for part in [system, user, end] if part]
    else:
        parts = [part for part in [system, user, assistant, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
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


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int,
    dataset: Dataset,
    for_validation: bool = False,
):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(
        create_prompt_formats, fn_kwargs={"for_validation": for_validation}
    )

    # Apply preprocessing to each batch of the dataset
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(
        lambda sample: len(sample["input_ids"]) < max_length
    )

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get LoRA module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        # Training parameters from here:
        # https://huggingface.co/blog/Llama2-for-non-engineers
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            weight_decay=0.01,
            warmup_ratio=0.1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            save_steps=10000,
            output_dir=output_dir,
            optim="adamw_torch",  # "paged_adamw_32bit",
            lr_scheduler_type="linear",
            num_train_epochs=15,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_file",
        "-t",
        dest="train_data_file",
        type=str,
        help="Training JSON data file.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        "-m",
        dest="model_name",
        type=str,
        help="Name of base model to train an adapter for.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        type=str,
        help="Path to save the trained adapter model.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        dest="cache_dir",
        type=str,
        default=None,
        required=False,
        help="The cache directory to save the downloaded models.",
    )

    args = parser.parse_args()
    model_name = args.model_name
    training_data_file = args.train_data_file
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    hf_token = os.getenv("HF_ACCESS_TOKEN")

    dataset = load_dataset(
        "json",
        data_files=training_data_file,
        split="train",
    )
    print(f"Number of prompts: {len(dataset)}")
    print(f"Column names are: {dataset.column_names}")

    bnb_config = create_bnb_config()

    model, tokenizer = load_model(model_name, bnb_config, cache_dir, hf_token)

    max_length = get_max_length(model)

    dataset = preprocess_dataset(tokenizer, max_length, 0, dataset)
    train(model, tokenizer, dataset, output_dir)
