import json
import os
import sys

import pandas as pd

from joblib import delayed, Parallel
from tqdm import tqdm


def create_label_dict(label_df, json_directory):

    label_dict = {}

    for idx in range(label_df.shape[0]):
        source = label_df.iloc[idx]["source"]
        label = label_df.iloc[idx]["label"]
        json_filename = os.path.join(json_directory, source) + ".json"

        if os.path.exists(json_filename):
            label_dict[source] = label

    return label_dict


def create_dataframe(json_directory, csv_directory, source_name, label_value):

    article_id = []
    collection_utc = []
    publication_utc = []
    date_of_publication = []
    date = []
    source = []
    author = []
    title = []
    content = []
    url = []

    csv_filename = os.path.join(csv_directory, source_name) + ".csv"
    json_filename = os.path.join(json_directory, source_name) + ".json"

    news_data = json.load(open(json_filename))
    for data in news_data:
        article_id.append(data["id"])
        collection_utc.append(data["collection_utc"])
        publication_utc.append(data["published_utc"])
        date_of_publication.append(data["published"])
        date.append(data["date"])
        source.append(data["source"])
        author.append(data["author"])
        title.append(data["title"])
        content.append(data["content"])
        url.append(data["url"])

    dataframe = pd.DataFrame()
    dataframe["id"] = article_id
    dataframe["collection_utc"] = collection_utc
    dataframe["publication_utc"] = publication_utc
    dataframe["date_of_publication"] = date_of_publication
    dataframe["date"] = date
    dataframe["source"] = source
    dataframe["author"] = author
    dataframe["title"] = title
    dataframe["content"] = content
    dataframe["label"] = [label_value for _ in range(dataframe.shape[0])]
    dataframe["url"] = url

    dataframe.to_csv(csv_filename, index=False)


def prepare_dataframe(json_directory, jobs, csv_directory, label_dict):

    _ = Parallel(n_jobs=jobs, backend="threading")(
        delayed(create_dataframe)(
            json_directory=json_directory,
            csv_directory=csv_directory,
            source_name=source,
            label_value=label,
        )
        for source, label in tqdm(label_dict.items())
    )


def merge_dataframe(
    csv_directory,
    csv_path,
):

    dataframe = pd.DataFrame()

    for csv_filename in tqdm(os.listdir(csv_directory)):

        csv_filename = os.path.join(csv_directory, csv_filename)
        df = pd.read_csv(csv_filename)
        dataframe = dataframe.append(df, ignore_index=True)

    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    dataframe.to_csv(csv_path, index=False)


def main():
    if len(sys.argv) != 2:
        print("Pass config file as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    json_directory = config["utils"]["jsonDirectory"]
    jobs = config["utils"]["numberofJobs"]

    csv_directory = config["csv"]["csvDirectory"]
    label_csv_path = config["csv"]["labelPath"]
    csv_path = config["csv"]["dataframePath"]

    label_df = pd.read_csv(label_csv_path)

    label_dict = create_label_dict(label_df=label_df, json_directory=json_directory)
    prepare_dataframe(
        json_directory=json_directory,
        jobs=jobs,
        csv_directory=csv_directory,
        label_dict=label_dict,
    )
    merge_dataframe(
        csv_directory=csv_directory,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
    print(
        "\n--------------------\nCreated train, test and val dataframes!\n--------------------\n"
    )
