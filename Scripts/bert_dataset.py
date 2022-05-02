import torch
from torch.utils.data import Dataset


class FakeNewsDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        column_name,
        sequence_length,
    ):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.column_name = column_name
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        text = self.dataframe.iloc[idx][self.column_name]
        text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        label = [self.dataframe.iloc[idx]["label"]]
        label = torch.FloatTensor(label).type(torch.LongTensor)
        return text, label
