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

    train_csv_path = config["csv"]["trainPath"]
    val_csv_path = config["csv"]["valPath"]

    title_flag = config["dataset"]["titleFlag"]
    sequence_length = config["dataset"]["sequenceLength"]
    batch_size = config["dataset"]["batchSize"]
    shuffle_flag = config["dataset"]["shuffleFlag"]
    num_workers = config["dataset"]["numWorkers"]

    model_name = config["model"]["modelName"]
    hidden_units = config["model"]["hiddenUnits"]
    dropout = config["model"]["dropoutValue"]

    learning_rate = config["train"]["learningRate"]
    continue_flag = config["train"]["continueFlag"]
    continue_checkpoint_path = config["train"]["checkpointToBeContinued"]

    start_epoch = config["train"]["startEpoch"]
    end_epoch = config["train"]["endEpoch"]
    logs_path = config["train"]["logsPath"]
    checkpoint_dir = config["train"]["checkpointDir"]

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if title_flag:
        column_name = "preprocessed_title"
        print_column_name = "Title"
    else:
        column_name = "preprocessed_content"
        print_column_name = "Content"

    print("Using {} for classification!".format(print_column_name.lower()))

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = FakeNewsDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        column_name=column_name,
        sequence_length=sequence_length,
    )
    params = {
        "batch_size": batch_size,
        "shuffle": shuffle_flag,
        "num_workers": num_workers,
    }
    train_gen = DataLoader(train_dataset, **params)

    val_dataset = FakeNewsDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        column_name=column_name,
        sequence_length=sequence_length,
    )
    params = {
        "batch_size": 1,
        "shuffle": shuffle_flag,
        "num_workers": num_workers,
    }
    val_gen = DataLoader(val_dataset, **params)

    model = BERTModel(model_name=model_name, hidden_units=hidden_units, dropout=dropout)

    model, criterion, optimizer = prepare_model_for_training(
        model=model,
        learning_rate=learning_rate,
        continue_flag=continue_flag,
        continue_checkpoint_path=continue_checkpoint_path,
    )

    print(model)

    train_model(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        train_gen=train_gen,
        val_gen=val_gen,
        logs_path=logs_path,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
    print("\n--------------------\nTraining complete!\n--------------------\n")
