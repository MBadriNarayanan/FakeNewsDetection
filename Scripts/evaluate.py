import json
import sys
import torch

import pandas as pd

from torch.utils.data import DataLoader

from dataset import *
from model import *
from utils import *


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

    title_dict_path = config["dataset"]["titleDictPath"]
    title_tokenizer_path = config["dataset"]["titleTokenizerPath"]
    content_dict_path = config["dataset"]["contentDictPath"]
    content_tokenizer_path = config["dataset"]["contentTokenizerPath"]
    oov_token = config["dataset"]["oovToken"]
    pad_token = config["dataset"]["paddingToken"]
    padding = config["dataset"]["paddingMode"]
    embed_dim = config["dataset"]["embeddingDimension"]
    shuffle_flag = config["dataset"]["shuffleFlag"]
    num_workers = config["dataset"]["numWorkers"]
    batch_size = 1

    encoder_dim = embed_dim
    encoder_ffunits = config["model"]["encoderFeedForwardUnits"]
    encoder_layers = config["model"]["encoderLayers"]
    encoder_heads = config["model"]["encoderHeads"]
    dropout = config["model"]["dropoutValue"]
    activation = config["model"]["activationFunction"]
    title_sequence_size = config["model"]["titleSequenceSize"]
    content_sequence_size = config["model"]["contentSequenceSize"]
    embedding_flag = config["model"]["embeddingFlag"]
    title_embedding_matrix = config["model"]["titleEmbeddingMatrixPath"]
    content_embedding_matrix = config["model"]["contentEmbeddingMatrixPath"]
    hidden_units_1 = config["model"]["hiddenUnits1"]
    hidden_units_2 = config["model"]["hiddenUnits2"]
    hidden_units_3 = config["model"]["hiddenUnits3"]
    information_vector_dimension = config["model"]["informationVectorDimension"]

    checkpoint_path = config["evaluate"]["checkpointPath"]
    report_path = config["evaluate"]["reportPath"]

    test_df = pd.read_csv(test_csv_path)

    data_dict = create_word_dict(
        title_dict_path=title_dict_path,
        title_tokenizer_path=title_tokenizer_path,
        content_dict_path=content_dict_path,
        content_tokenizer_path=content_tokenizer_path,
        df="",
        oov_token=oov_token,
        pad_token=pad_token,
    )

    title_word_dict = data_dict["title_word_dict"]
    title_tokenizer = data_dict["title_tokenizer"]
    content_word_dict = data_dict["content_word_dict"]
    content_tokenizer = data_dict["content_tokenizer"]

    title_vocab_size = len(title_word_dict)
    content_vocab_size = len(content_word_dict)

    test_dataset = FakeNewsDataset(
        dataframe=test_df,
        title_word_dict=title_word_dict,
        title_tokenizer=title_tokenizer,
        content_word_dict=content_word_dict,
        content_tokenizer=content_tokenizer,
        title_sequence_length=title_sequence_size,
        content_sequence_length=content_sequence_size,
        padding=padding,
        embed_dim=embed_dim,
    )
    params = {
        "batch_size": batch_size,
        "shuffle": shuffle_flag,
        "num_workers": num_workers,
    }
    test_gen = DataLoader(test_dataset, **params)

    model = FakeNewsTransformer(
        title_vocab_size=title_vocab_size,
        content_vocab_size=content_vocab_size,
        encoder_dim=encoder_dim,
        encoder_ffunits=encoder_ffunits,
        encoder_layers=encoder_layers,
        encoder_heads=encoder_heads,
        dropout=dropout,
        activation=activation,
        title_sequence_size=title_sequence_size,
        content_sequence_size=content_sequence_size,
        embedding_flag=embedding_flag,
        title_embedding_matrix=title_embedding_matrix,
        content_embedding_matrix=content_embedding_matrix,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        hidden_units_3=hidden_units_3,
        information_vector_dimension=information_vector_dimension,
        device=device,
    ).to(device)

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
