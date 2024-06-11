import argparse
import pickle
import ndjson
import json
import tiktoken
from prompt_message import (
    system_message,
    generate_user_message,
    generate_assistant_message,
)

def save_data_split(data: list, tokens: list, encoding_model: str, save_file: str) -> None:
    """Save data to file.

    Args:
        tokens: list of strings to save
        encoding_model: the name of the model whose encoding we need to use.
            Will be used as: tiktoken.encoding_for_model(encoding_model)
        save_file: path to save the file
    """
    print(f"Saving data split: {len(tokens)} : {save_file}")

    encoding = tiktoken.encoding_for_model(encoding_model)

    num_language_tokens = 0
    num_system_tokens = 0
    num_user_tokens = 0
    num_assistant_tokens = 0

    traj_only = False

    train_messages = []
    num_samples = len(tokens)
    for token_i, token in enumerate(tokens):
        if token_i >= num_samples:
            break
        user_message = generate_user_message(data, token)
        assitant_message = generate_assistant_message(data, token, traj_only=traj_only)
        num_language_tokens += len(encoding.encode(system_message))
        num_system_tokens += len(encoding.encode(system_message))
        num_language_tokens += len(encoding.encode(user_message))
        num_user_tokens += len(encoding.encode(user_message))
        num_language_tokens += len(encoding.encode(assitant_message))
        num_assistant_tokens += len(encoding.encode(assitant_message))

        train_message = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assitant_message},
            ],
            "token": token,
        }
        train_messages.append(train_message)

    print("#### Cost Summarization ####")
    print(f"Number of system tokens: {num_system_tokens}")
    print(f"Number of user tokens: {num_user_tokens}")
    print(f"Number of assistant tokens: {num_assistant_tokens}")
    print(f"Number of total tokens: {num_language_tokens}")

    with open(save_file, "w", encoding="utf-8") as f:
        ndjson.dump(train_messages, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./create_data_split.py")
    parser.add_argument(
        '--model_name', '-m',
        dest="model_name",
        type=str,
        help = "Model name, used for encoding the messages.",
        required=True
    )
    parser.add_argument(
        '--data_file', '-d',
        dest="data_file",
        type=str,
        help="JSON data file. Will be split to train and validation.",
        required=True
    )
    parser.add_argument(
        '--split_data_file', '-s',
        dest="split_data_file",
        type=str,
        help="File containing data split.",
        required=True
    )
    parser.add_argument(
        '--val_data_file', '-v',
        dest="val_data_file",
        type=str,
        help="Path to save validation JSON data file",
        required=True
    )
    parser.add_argument(
        '--train_data_file', '-t',
        dest="train_data_file",
        type=str,
        help="Path to save validation JSON data file",
        required=True
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    model_name = FLAGS.model_name
    data_file = FLAGS.data_file
    split_data_file = FLAGS.split_data_file
    val_data_file = FLAGS.val_data_file
    train_data_file = FLAGS.train_data_file

    # Load data and split ids
    data = pickle.load(open(data_file, "rb"))
    split = json.load(open(split_data_file, "r", encoding="utf-8"))

    # Get training and validation tokens from split
    train_tokens = split["train"]
    val_tokens = split["val"]

    save_data_split(data, train_tokens, model_name, train_data_file)
    save_data_split(data, val_tokens, model_name, val_data_file)
