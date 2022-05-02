import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from utils import load_pickle_file, save_pickle_file


def create_word_dict(
    df,
    column_name,
    print_column_name,
    oov_token,
    pad_token,
    tokenizer_path,
    word_dict_path,
):

    try:

        tokenizer = load_pickle_file(tokenizer_path)
        word_dict = load_pickle_file(word_dict_path)
        print(
            "{} tokenizer and word dict loaded from memory!".format(print_column_name)
        )

    except:

        tokenizer = Tokenizer(oov_token=oov_token)
        tokenizer.fit_on_texts(df[column_name])
        word_dict = tokenizer.word_index
        word_dict[pad_token] = 0
        word_dict = {v: k for k, v in word_dict.items()}
        save_pickle_file(tokenizer, tokenizer_path, True)
        save_pickle_file(word_dict, word_dict_path)
        print("Created {} tokenizer and word dict!".format(print_column_name.lower()))

    return {"tokenizer": tokenizer, "word_dict": word_dict}


def create_embedding_matrix(word_dict, embed_dim, embed_index, matrix_path):

    max_words = len(word_dict)
    embedding_matrix = np.zeros((max_words, embed_dim))
    for idx, word in word_dict.items():
        if idx < max_words:
            embed_vector = embed_index.get(word)
        if embed_vector is not None:
            embedding_matrix[idx] = embed_vector
    save_pickle_file(data=embedding_matrix, file_path=matrix_path)
    return embedding_matrix


def generate_sequence(tokenizer, df, column_name, sequence_length, padding):

    sequence = tokenizer.texts_to_sequences(df[column_name])
    sequence = pad_sequences(sequence, maxlen=sequence_length, padding=padding)
    label = np.array(df["label"])
    label = to_categorical(label, 3)

    return sequence, label


def get_embedding_matrix(
    embedding_matrix_path,
    print_column_name,
    vector_path,
    word_dict,
    embed_dim,
):

    try:

        embedding_matrix = load_pickle_file(embedding_matrix_path)
        print("{} embedding matrix loaded from memory!".format(print_column_name))

    except:

        embed_index = dict()
        vector_file = open(vector_path, encoding="utf8")
        for line in vector_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embed_index[word] = coefs
        vector_file.close()

        embedding_matrix = create_embedding_matrix(
            word_dict=word_dict,
            embed_dim=embed_dim,
            embed_index=embed_index,
            matrix_path=embedding_matrix_path,
        )
        print("Created {} embedding matrix!".format(print_column_name.lower()))

    return embedding_matrix


def loss_plot(history, image_path):

    plt.plot(history.history["loss"], label="Training data")
    plt.plot(history.history["val_loss"], label="Validation data")
    plt.title("Loss")
    plt.ylabel("Loss value")
    plt.xlabel("No. epoch")
    plt.legend(loc="upper left")
    plt.savefig(image_path)
    plt.show()


def train_model(
    model,
    loss,
    optimizer,
    X_train,
    y_train,
    epochs,
    X_val,
    y_val,
    batch_size,
    callbacks,
    image_path,
):

    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )
    loss_plot(history=history, image_path=image_path)


def evaluate_model(model, checkpoint_path, X_test, y_test, threshold, report_path):

    model.load_weights(filepath=checkpoint_path)
    print("Loaded checkpoint:", checkpoint_path, "for evaluation!")
    model.summary()
    y_pred = model.predict(X_test)
    upper, lower = 1, 0
    y_pred = np.where(y_pred > threshold, upper, lower)
    y_test = np.where(y_test > threshold, upper, lower)

    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 3)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Reliable", "Mixed", "Unreliable"],
        digits=3,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, y_pred)

    report_file = open(report_path, "w")
    report_file.write("Metrics for the checkpoint: {}\n".format(checkpoint_path))
    report_file.write("Accuracy: {}\n".format(accuracy))
    report_file.write("Classification Report\n")
    report_file.write("{}\n".format(report))
    report_file.write("Confusion Matrix\n")
    report_file.write("{}\n".format(matrix))
    report_file.write("-----------------------------\n")
    report_file.close()
