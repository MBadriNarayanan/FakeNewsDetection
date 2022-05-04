import json
import sys
import torch
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

    train_csv_path = config["csv"]["trainPath"]
    val_csv_path = config["csv"]["valPath"]

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

    checkpoint_monitor = config["train"]["checkpointMonitor"]
    paitence_value = config["train"]["paitenceValue"]
    checkpoint_mode = config["train"]["checkpointMode"]
    checkpoint_path = config["train"]["checkpointPath"]
    loss = config["train"]["lossFunction"]
    optimizer = config["train"]["optimizerFunction"]
    epochs = config["train"]["numberOfEpochs"]
    batch_size = config["train"]["batchSize"]
    loss_image_path = config["train"]["lossImagePath"]

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if title_flag:
        column_name = "preprocessed_title"
        print_column_name = "Title"
    else:
        column_name = "preprocessed_content"
        print_column_name = "Content"

    data_dict = create_word_dict(
        df=train_df,
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

    X_train, y_train = generate_sequence(
        tokenizer=tokenizer,
        df=train_df,
        column_name=column_name,
        sequence_length=sequence_length,
        padding=padding,
    )

    X_val, y_val = generate_sequence(
        tokenizer=tokenizer,
        df=val_df,
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

    early_stopping = EarlyStopping(
        monitor=checkpoint_monitor,
        patience=paitence_value,
        mode=checkpoint_mode,
    )
    save_best = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
    )
    callbacks = [early_stopping, save_best]

    train_model(
        model=recurrent_model,
        loss=loss,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        epochs=epochs,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        callbacks=callbacks,
        image_path=loss_image_path,
    )


if __name__ == "__main__":
    main()
    print("\n--------------------\nTraining complete!\n--------------------\n")
