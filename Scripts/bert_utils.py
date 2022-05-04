import torch
import torch.cuda

from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim import Adam

from utils import get_model_parameters


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

            text = gen_values[0].to(device)
            target = gen_values[1].to(device)
            target = target.reshape(target.shape[0])

            input_id = text["input_ids"].squeeze(1)
            mask = text["attention_mask"]

            optimizer.zero_grad()
            y_hat = model(input_id, mask)
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

            del text, target
            del input_id, mask
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

                text = gen_values[0].to(device)
                target = gen_values[1].to(device)
                target = target.reshape(target.shape[0])

                input_id = text["input_ids"].squeeze(1)
                mask = text["attention_mask"]

                y_hat = model(input_id, mask)
                loss = criterion(y_hat, target)
                val_loss += loss.item()

                y_pred = log_softmax(y_hat, dim=1)
                y_pred = torch.argmax(y_pred, dim=1)
                y_pred = y_pred.detach().cpu().tolist()[0]
                target = target.detach().cpu().tolist()[0]

                ground_truth.append(target)
                prediction.append(y_pred)

                del text, target
                del input_id, mask
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

            text = gen_values[0].to(device)
            target = gen_values[1].to(device)
            target = target.reshape(target.shape[0])

            input_id = text["input_ids"].squeeze(1)
            mask = text["attention_mask"]

            y_hat = model(input_id, mask)
            y_pred = log_softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().tolist()[0]
            target = target.detach().cpu().tolist()[0]

            ground_truth.append(target)
            prediction.append(y_pred)

            del text, target
            del input_id, mask
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
