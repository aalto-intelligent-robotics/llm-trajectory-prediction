import argparse
import numpy as np
import ndjson
import re
import ast
from datasets import load_dataset, Dataset


def extract(input_string):
    # Define the pattern for extracting numeric values
    pattern = r"\d+\.\d+"

    # Use regular expression to find all numeric values in the trajectory string
    matches = re.findall(pattern, input_string)

    # Convert the matched values to a list of tuples
    trajectory_list = [
        (float(matches[i]), float(matches[i + 1]))
        for i in range(0, len(matches), 2)
    ]

    return trajectory_list


def save_to_text_file(file_path, data):
    with open(file_path, "w") as text_file:
        for item in data:
            text_file.write(str(item) + "\n")


def calc_l2(traj1: list, traj2: list, first_n_pairs: int = -1) -> {}:
    """Calculate L2 loss between two given trajectories.
    Args:
        traj1: list of (x,y) points in trajectory 1
        traj2: list of (x,y) points in trajectory 2
        first_n_pairs: int, the number of pairs from both trajectories
            to include in the evaluation
            For example, 6 pairs corresponds to 3 seconds,
            4 pairs - to two seconds, 2 pairs - to 1 second

    Returns:
        L2 value
    """
    if first_n_pairs <= 0:
        first_n_pairs = min(len(traj1), len(traj2))

    l2 = 0.0
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(traj1, traj2)):
        if i >= first_n_pairs:
            break
        l2 += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # l2 = l2/(min(len(traj1), len(traj2)))
    l2 = l2 / first_n_pairs  # (min(len(traj1), len(traj2)))

    return l2


def extract_trajectory(data, result_file):
    """Extract trajectory list from given message text.

    Args:
        text: The mssage text. Expected is an assistant message and Trajectory:

    Returns:
        Numpy float array containing the trajectory
    """
    count = 0
    bad_response = []
    total = 0
    data_dict = {}
    for message_instance in data:
        if message_instance["token"] not in data_dict:
            data_dict[message_instance["token"]] = []

    for i, message_instance in enumerate(data):
        try:
            predicted_message = message_instance["GPT"].split("\n")
            for ii, string in enumerate(predicted_message):
                if "Trajectory" in string:
                    # if 'Historical' in string:
                    #     # pass
                    #     print(string)
                    # else:
                    if not "Historical" in string:
                        ind = ii
                        flag = True
                        # print(predicted_message)

                        try:
                            extract_traj = ast.literal_eval(
                                predicted_message[ind + 1]
                            )
                            data_dict[data[i]["token"]].append(
                                {"traj": extract_traj}
                            )
                            total = total + 1
                        except:
                            bad_response.append([str(i), message_instance["GPT"]])
                            count = count + 1

            gt_message = message_instance["GT"]
        except:
            # print(predicted_message)
            count = count + 1

    for message_instance in data:
        gt_message = message_instance["GT"].split("\n")
        for ik, string in enumerate(gt_message):
            if "Trajectory" in string:
                ind_g = ik
                break  # Stop searching after the first occurrence
        traj_gt = ast.literal_eval(gt_message[ind_g + 1])
        # print(gt_message)
        data_dict[message_instance["token"]].append({"traj_gt": traj_gt})

    save_to_text_file(result_file, data_dict.values())
    print(f"Data saved to {result_file}")
    print("False prediction:", count, total)
    return data_dict


def score_cal(data_dict):
    """Create and save evaluations for ground-truth and predicted data."""
    eval_3 = []
    eval_2 = []
    eval_1 = []
    for values_list in data_dict.values():
        if len(values_list) == 2:
            pred_traj = values_list[0]["traj"]
            gt_traj = values_list[1]["traj_gt"]
            # print(f"pred_traj: {pred_traj}")
            # print(f"gt_traj: {gt_traj}")
            if len(pred_traj) == 6 and isinstance(pred_traj, list):
                eval_3.append(calc_l2(gt_traj, pred_traj, 6))
                eval_2.append(calc_l2(gt_traj, pred_traj, 4))
                eval_1.append(calc_l2(gt_traj, pred_traj, 2))
            else:
                pass
                # print(f"ERROR: {pred_traj}")

    avg_3 = np.sum(np.array(eval_3)) / len(eval_3)
    avg_2 = np.sum(np.array(eval_2)) / len(eval_2)
    avg_1 = np.sum(np.array(eval_1)) / len(eval_1)
    print("L2_3sec", avg_3)
    print("L2_2sec", avg_2)
    print("L2_1sec", avg_1)
    print("L2_avg", (avg_1 + avg_2 + avg_3) / 3)
    return avg_3, avg_2, avg_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file",
        "-p",
        dest="prediction_file",
        type=str,
        help="Output Predictions JSON Data File",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        "-o",
        dest="output_path",
        type=str,
        required=True,
    )
    FLAGS, unparsed = parser.parse_known_args()
    prediction_file = FLAGS.prediction_file
    output_path = FLAGS.output_path

    with open(prediction_file, "r", encoding="utf-8") as file:
        # Load the JSON data into a Python dictionary
        data_output = ndjson.load(file)
    print(f"Pred instances: {len(data_output)}")

    result = extract_trajectory(data_output, output_path)
    L2 = score_cal(result)
    