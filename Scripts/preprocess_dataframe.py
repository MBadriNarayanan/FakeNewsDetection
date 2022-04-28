import json
import re
import nltk
import string
import sys

import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

tqdm.pandas()
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
punctuations = string.punctuation
remove_punct = str.maketrans("", "", punctuations)


def preprocess_title(title):
    title = title.lower()
    title = title.replace("&apos;", "'")
    title = re.sub(r"\d+", "NUM", title)
    title = title.translate(remove_punct)
    title = re.sub("((www.[^s]+)|(https?://[^s]+))", " ", title)
    title = re.sub("\W+", " ", title)
    title = re.sub("\s\s+", " ", title)
    return title


def preprocess_content(
    content,
    summarization_model,
    summarization_tokenizer,
    return_tensors,
    max_tokens,
    truncation_flag,
    max_length,
    min_length,
    length_penalty,
    num_beams,
    early_stopping_flag,
):
    inputs = summarization_tokenizer(
        "summarize: " + content,
        return_tensors=return_tensors,
        max_length=max_tokens,
        truncation=truncation_flag,
    )
    outputs = summarization_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping=early_stopping_flag,
    )
    summarized_content = summarization_tokenizer.decode(outputs[0])
    summarized_content = summarized_content.lower()
    content = summarized_content[5:].strip()
    content = content[:-4].strip()
    content = preprocess_title(content)
    return content


def preprocess_data(
    title,
    content,
    summarization_model,
    summarization_tokenizer,
    return_tensors,
    max_tokens,
    truncation_flag,
    max_length,
    min_length,
    length_penalty,
    num_beams,
    early_stopping_flag,
):

    title = preprocess_title(title=title)
    content = preprocess_content(
        content=content,
        summarization_model=summarization_model,
        summarization_tokenizer=summarization_tokenizer,
        return_tensors=return_tensors,
        max_tokens=max_tokens,
        truncation_flag=truncation_flag,
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping_flag=early_stopping_flag,
    )

    return title, content


def preprocess_dataframe(
    csv_path,
    sample_size,
    preprocessed_csv_path,
    train_csv_path,
    val_csv_path,
    test_csv_path,
    val_size,
    test_size,
    random_state,
    summarization_model,
    summarization_tokenizer,
    return_tensors,
    max_tokens,
    truncation_flag,
    content_max_length,
    content_min_length,
    length_penalty,
    num_beams,
    early_stopping_flag,
):

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["title", "content"])
    df = df.drop_duplicates(subset=["title", "content"])
    _, sample_df = train_test_split(
        df, test_size=sample_size, stratify=df["label"], random_state=random_state
    )
    sample_df = sample_df.reset_index(drop=True)

    sample_df["preprocessed_data"] = sample_df.progress_apply(
        lambda row: preprocess_data(
            title=row["title"],
            content=row["content"],
            summarization_model=summarization_model,
            summarization_tokenizer=summarization_tokenizer,
            return_tensors=return_tensors,
            max_tokens=max_tokens,
            truncation_flag=truncation_flag,
            max_length=content_max_length,
            min_length=content_min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping_flag=early_stopping_flag,
        ),
        axis=1,
    )

    sample_df["preprocessed_title"] = sample_df["preprocessed_data"].apply(
        lambda row: row[0]
    )
    sample_df["preprocessed_content"] = sample_df["preprocessed_data"].apply(
        lambda row: row[1]
    )

    sample_df = sample_df[
        [
            "id",
            "collection_utc",
            "publication_utc",
            "date_of_publication",
            "date",
            "source",
            "author",
            "url",
            "title",
            "preprocessed_title",
            "content",
            "preprocessed_content",
            "label",
        ]
    ]

    sample_df.to_csv(preprocessed_csv_path, index=False)

    train_df, test_df = train_test_split(
        sample_df,
        test_size=test_size,
        stratify=sample_df["label"],
        random_state=random_state,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)


def main():
    if len(sys.argv) != 2:
        print("Pass config file as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    csv_path = config["csv"]["csvPath"]
    sample_size = config["csv"]["sampleSize"]
    preprocessed_csv_path = config["csv"]["preprocessedCSVPath"]
    val_size = config["csv"]["valSize"]
    test_size = config["csv"]["testSize"]
    random_state = config["csv"]["randomState"]
    preprocessed_train_csv_path = config["csv"]["preprocessedTrainPath"]
    preprocessed_val_csv_path = config["csv"]["preprocessedValPath"]
    preprocessed_test_csv_path = config["csv"]["preprocessedTestPath"]

    summarization_model_name = config["summarization"]["modelName"]
    return_tensors = config["summarization"]["returnTensors"]
    max_tokens = config["summarization"]["maxTokens"]
    truncation_flag = config["summarization"]["truncationFlag"]
    content_max_length = config["summarization"]["contentMaxLength"]
    content_min_length = config["summarization"]["contentMinLength"]
    length_penalty = config["summarization"]["lengthPenalty"]
    num_beams = config["summarization"]["numberOfBeams"]
    early_stopping_flag = config["summarization"]["earlystoppingFlag"]

    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
        summarization_model_name
    )
    summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)

    preprocess_dataframe(
        csv_path=csv_path,
        sample_size=sample_size,
        preprocessed_csv_path=preprocessed_csv_path,
        train_csv_path=preprocessed_train_csv_path,
        val_csv_path=preprocessed_val_csv_path,
        test_csv_path=preprocessed_test_csv_path,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        summarization_model=summarization_model,
        summarization_tokenizer=summarization_tokenizer,
        return_tensors=return_tensors,
        max_tokens=max_tokens,
        truncation_flag=truncation_flag,
        content_max_length=content_max_length,
        content_min_length=content_min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping_flag=early_stopping_flag,
    )


if __name__ == "__main__":
    main()
    print(
        "\n--------------------\nPreprocess of train, test and val dataframes complete!\n--------------------\n"
    )
