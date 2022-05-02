import json
import sys
import torch

import pandas as pd

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from bert_dataset import *
from bert_model import *
from bert_utils import *


def main():
    if len(sys.argv) != 2:
        print("Pass config file of model as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available!")
    else:
        device = torch.device("cpu")
        print("GPU not available!")

    test_csv_path = config["csv"]["testPath"]

    title_flag = config["dataset"]["titleFlag"]
    sequence_length = config["dataset"]["sequenceLength"]
    shuffle_flag = config["dataset"]["shuffleFlag"]
    num_workers = config["dataset"]["numWorkers"]

    model_name = config["model"]["modelName"]
    hidden_units = config["model"]["hiddenUnits"]
    dropout = config["model"]["dropoutValue"]

    checkpoint_path = config["evaluate"]["checkpointPath"]
    report_path = config["evaluate"]["reportPath"]

    test_df = pd.read_csv(test_csv_path)

    if title_flag:
        column_name = "preprocessed_title"
        print_column_name = "Title"
    else:
        column_name = "preprocessed_content"
        print_column_name = "Content"

    print("Using {} for classification!".format(print_column_name.lower()))

    tokenizer = BertTokenizer.from_pretrained(model_name)

    test_dataset = FakeNewsDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        column_name=column_name,
        sequence_length=sequence_length,
    )
    params = {
        "batch_size": 1,
        "shuffle": shuffle_flag,
        "num_workers": num_workers,
    }
    test_gen = DataLoader(test_dataset, **params)

    model = BERTModel(model_name=model_name, hidden_units=hidden_units, dropout=dropout)
    model = prepare_model_for_evaluation(model=model, checkpoint_path=checkpoint_path)

    print(model)

    evaluate_model(
        model=model,
        device=device,
        test_gen=test_gen,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
    print("\n--------------------\nEvaluation complete!\n--------------------\n")
