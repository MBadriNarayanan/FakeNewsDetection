import json
import sys
import torch
import pandas as pd
from recurrent_model import Model
from recurrent_utils import *


def main():
    if len(sys.argv) != 2:
        print("Pass config file of model as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    if torch.cuda.is_available():
        print("GPU available!")
    else:
        print("GPU not available!")

    test_csv_path = config["csv"]["testPath"]

    title_flag = config["dataset"]["titleFlag"]
    word_dict_path = config["dataset"]["dictPath"]
    tokenizer_path = config["dataset"]["tokenizerPath"]
    sequence_length = config["dataset"]["sequenceLength"]
    oov_token = config["dataset"]["oovToken"]
    pad_token = config["dataset"]["paddingToken"]
    padding = config["dataset"]["paddingMode"]

    embedding_matrix_path = config["vector"]["embeddingMatrixPath"]
    vector_path = config["vector"]["vectorPath"]

    embedding_flag = config["model"]["embeddingFlag"]
    embed_dim = config["model"]["embeddingDimension"]
    model_flag = config["model"]["modelFlag"]
    bidirectional_flag = config["model"]["biDirectionalFlag"]
    dropout = config["model"]["dropoutValue"]
    recurrent_units = config["model"]["recurrentUnits"]
    hidden_units = config["model"]["hiddenUnits"]

    checkpoint_path = config["evaluate"]["checkpointPath"]
    threshold = config["evaluate"]["thresholdValue"]
    report_path = config["evaluate"]["reportPath"]

    test_df = pd.read_csv(test_csv_path)

    if title_flag:
        column_name = "preprocessed_title"
        print_column_name = "Title"
    else:
        column_name = "preprocessed_content"
        print_column_name = "Content"

    data_dict = create_word_dict(
        df=None,
        column_name=column_name,
        print_column_name=print_column_name,
        oov_token=oov_token,
        pad_token=pad_token,
        tokenizer_path=tokenizer_path,
        word_dict_path=word_dict_path,
    )

    word_dict = data_dict["word_dict"]
    tokenizer = data_dict["tokenizer"]

    vocab_size = len(word_dict)

    X_test, y_test = generate_sequence(
        tokenizer=tokenizer,
        df=test_df,
        column_name=column_name,
        sequence_length=sequence_length,
        padding=padding,
    )

    if embedding_flag:

        embedding_matrix = get_embedding_matrix(
            embedding_matrix_path=embedding_matrix_path,
            print_column_name=print_column_name,
            vector_path=vector_path,
            word_dict=word_dict,
            embed_dim=embed_dim,
        )
        print("Using pretrained glove embeddings!")

        model = Model(
            model_flag=model_flag,
            bidirectional_flag=bidirectional_flag,
            embedding_flag=embedding_flag,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embedding_matrix=embedding_matrix,
            dropout=dropout,
            recurrent_units=recurrent_units,
            hidden_units=hidden_units,
        )

    else:

        print("Using custom embeddings!")

        model = Model(
            model_flag=model_flag,
            bidirectional_flag=bidirectional_flag,
            embedding_flag=embedding_flag,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embedding_matrix=None,
            dropout=dropout,
            recurrent_units=recurrent_units,
            hidden_units=hidden_units,
        )

    recurrent_model = model.create_model()

    evaluate_model(
        model=recurrent_model,
        checkpoint_path=checkpoint_path,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
    print("\n--------------------\nEvaluation complete!\n--------------------\n")
