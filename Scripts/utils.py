import pickle
import torch
import torch.cuda

from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.nn.init import xavier_uniform_
from torch.optim import Adam


def save_pickle_file(data, file_path, flag=False):
    if flag:
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)


def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_word_dict(
    title_dict_path,
    title_tokenizer_path,
    content_dict_path,
    content_tokenizer_path,
    df,
    oov_token,
    pad_token,
):
    try:

        title_word_dict = load_pickle_file(file_path=title_dict_path)
        title_tokenizer = load_pickle_file(file_path=title_tokenizer_path)
        content_word_dict = load_pickle_file(file_path=content_dict_path)
        content_tokenizer = load_pickle_file(file_path=content_tokenizer_path)

        print("Word dictionaries and tokenizers loaded from memory!")

    except:

        title_tokenizer = Tokenizer(oov_token=oov_token)
        title_tokenizer.fit_on_texts(df["preprocessed_title"])
        title_word_dict = title_tokenizer.word_index
        title_word_dict[pad_token] = 0
        title_word_dict = {v: k for k, v in title_word_dict.items()}

        content_tokenizer = Tokenizer(oov_token=oov_token)
        content_tokenizer.fit_on_texts(df["preprocessed_content"])
        content_word_dict = content_tokenizer.word_index
        content_word_dict[pad_token] = 0
        content_word_dict = {v: k for k, v in content_word_dict.items()}

        save_pickle_file(data=title_word_dict, file_path=title_dict_path)
        save_pickle_file(
            data=title_tokenizer, file_path=title_tokenizer_path, flag=True
        )
        save_pickle_file(data=content_word_dict, file_path=content_dict_path)
        save_pickle_file(
            data=content_tokenizer, file_path=content_tokenizer_path, flag=True
        )

        print("Created word dictionaries and tokenizers!")

    data_dict = {
        "title_word_dict": title_word_dict,
        "title_tokenizer": title_tokenizer,
        "content_word_dict": content_word_dict,
        "content_tokenizer": content_tokenizer,
    }

    return data_dict


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


def get_embedding_matrix(
    title_embedding_matrix_path,
    content_embedding_matrix_path,
    vector_path,
    title_word_dict,
    content_word_dict,
    embed_dim,
):

    try:

        title_embedding_matrix = load_pickle_file(title_embedding_matrix_path)
        content_embedding_matrix = load_pickle_file(content_embedding_matrix_path)
        print("Loaded title and content embedding matrices from memory!")

    except:

        embed_index = dict()
        vector_file = open(vector_path, encoding="utf8")
        for line in vector_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embed_index[word] = coefs
        vector_file.close()

        title_embedding_matrix = create_embedding_matrix(
            word_dict=title_word_dict,
            embed_dim=embed_dim,
            embed_index=embed_index,
            matrix_path=title_embedding_matrix_path,
        )
        content_embedding_matrix = create_embedding_matrix(
            word_dict=content_word_dict,
            embed_dim=embed_dim,
            embed_index=embed_index,
            matrix_path=content_embedding_matrix_path,
        )
        print("Created title and content embedding matrices!")

    return title_embedding_matrix, content_embedding_matrix


def get_model_parameters(model):

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: ", params)
    print("--------------------------------------------")


def prepare_model_for_training(
    model, learning_rate, continue_flag, continue_checkpoint_path
):

    criterion = CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()

    optimizer = Adam(model.parameters(), lr=learning_rate)

    if continue_flag:

        print("Model loaded for further training!")
        checkpoint = torch.load(continue_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:

        print("Prepared model for training!")
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    get_model_parameters(model=model)

    return model, criterion, optimizer


def prepare_model_for_evaluation(model, checkpoint_path):

    if torch.cuda.is_available():
        model = model.cuda()

    print("Loaded checkpoint:", checkpoint_path, "for evaluation!")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("--------------------------------------------")

    return model


def train_model(
    model,
    device,
    criterion,
    optimizer,
    start_epoch,
    end_epoch,
    train_gen,
    val_gen,
    logs_path,
    checkpoint_dir,
):
    with open(logs_path, "at") as logs_file:
        logs_file.write(
            "Logs for the checkpoint stored at: {}/\n".format(checkpoint_dir)
        )

    for epoch in tqdm(range(start_epoch, end_epoch + 1)):
        train_loss = 0.0
        train_accuracy = 0.0
        model.train()

        for _, gen_values in enumerate(train_gen):

            title = gen_values[0].to(device)
            content = gen_values[1].to(device)
            target = gen_values[2].to(device)
            target = target.reshape(target.shape[0])

            optimizer.zero_grad()
            y_hat = model(title, content)
            loss = criterion(y_hat, target)
            batch_loss = loss.item()
            loss.backward()
            optimizer.step()

            y_pred = log_softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().tolist()
            target = target.detach().cpu().tolist()

            batch_accuracy = accuracy_score(target, y_pred)
            train_loss += batch_loss
            train_accuracy += batch_accuracy

            torch.cuda.empty_cache()

            del title, content, target
            del y_hat, loss, y_pred
            del batch_loss, batch_accuracy

        train_loss = train_loss / len(train_gen)
        train_loss = round(train_loss, 3)
        train_accuracy = train_accuracy / len(train_gen)
        train_accuracy = round(train_accuracy, 3)

        val_loss = 0.0
        ground_truth = []
        prediction = []

        with torch.no_grad():
            model.eval()

            for _, gen_values in enumerate(val_gen):

                title = gen_values[0].to(device)
                content = gen_values[1].to(device)
                target = gen_values[2].to(device)
                target = target.reshape(target.shape[0])

                y_hat = model(title, content)
                loss = criterion(y_hat, target)
                val_loss += loss.item()

                y_pred = log_softmax(y_hat, dim=1)
                y_pred = torch.argmax(y_pred, dim=1)
                y_pred = y_pred.detach().cpu().tolist()[0]
                target = target.detach().cpu().tolist()[0]

                ground_truth.append(target)
                prediction.append(y_pred)

                del title, content, target
                del y_hat, loss, y_pred

        val_loss = val_loss / len(val_gen)
        val_loss = round(val_loss, 3)
        val_accuracy = accuracy_score(ground_truth, prediction)
        val_accuracy = round(val_accuracy, 3)

        with open(logs_path, "at") as logs_file:
            logs_file.write(
                "Epoch: {}, Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}\n".format(
                    epoch, train_loss, train_accuracy, val_loss, val_accuracy
                )
            )
        ckpt_path = "{}/Epoch_{}.pt".format(checkpoint_dir, str(epoch))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            ckpt_path,
        )

        del train_loss, train_accuracy
        del ground_truth, prediction
        del val_loss, val_accuracy


def evaluate_model(
    model,
    device,
    test_gen,
    report_path,
    checkpoint_path,
):

    ground_truth = []
    prediction = []

    with torch.no_grad():
        model.eval()

        for _, gen_values in tqdm(enumerate(test_gen)):

            title = gen_values[0].to(device)
            content = gen_values[1].to(device)
            target = gen_values[2].to(device)
            target = target.reshape(target.shape[0])

            y_hat = model(title, content)
            y_pred = log_softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().tolist()[0]
            target = target.detach().cpu().tolist()[0]

            ground_truth.append(target)
            prediction.append(y_pred)

            del title, content, target
            del y_hat, y_pred

    accuracy = accuracy_score(ground_truth, prediction)
    accuracy = round(accuracy, 3)
    report = classification_report(
        ground_truth,
        prediction,
        digits=3,
        target_names=["Reliable", "Unreliable", "Mixed"],
        zero_division=0,
    )

    matrix = confusion_matrix(ground_truth, prediction)

    report_file = open(report_path, "w")
    report_file.write("Metrics for the checkpoint: {}\n".format(checkpoint_path))
    report_file.write("Accuracy: {}\n".format(accuracy))
    report_file.write("Classification Report\n")
    report_file.write("{}\n".format(report))
    report_file.write("Confusion Matrix\n")
    report_file.write("{}\n".format(matrix))
    report_file.write("-----------------------------\n")
    report_file.close()

    del ground_truth, prediction
    del accuracy, report
