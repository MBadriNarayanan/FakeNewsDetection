import torch

from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset


class FakeNewsDataset(Dataset):
    def __init__(
        self,
        dataframe,
        title_word_dict,
        title_tokenizer,
        content_word_dict,
        content_tokenizer,
        title_sequence_length,
        content_sequence_length,
        padding,
        embed_dim,
    ):
        self.dataframe = dataframe
        self.title_word_dict = title_word_dict
        self.title_tokenizer = title_tokenizer
        self.content_word_dict = content_word_dict
        self.content_tokenizer = content_tokenizer
        self.title_sequence_length = title_sequence_length
        self.content_sequence_length = content_sequence_length
        self.padding = padding
        self.embed_dim = embed_dim

    def __len__(self):
        return self.dataframe.shape[0]

    def _get_title_tensor(self, title):
        title_sequence = self.title_tokenizer.texts_to_sequences([title])
        title_sequence = pad_sequences(
            title_sequence, maxlen=self.title_sequence_length, padding=self.padding
        )
        title_sequence = title_sequence[0]
        title_sequence = [float(x) for x in title_sequence]
        title_tensor = torch.tensor(title_sequence).type(torch.LongTensor)
        return title_tensor

    def _get_content_tensor(self, content):
        content_sequence = self.content_tokenizer.texts_to_sequences([content])
        content_sequence = pad_sequences(
            content_sequence, maxlen=self.content_sequence_length, padding=self.padding
        )
        content_sequence = list(content_sequence[0])
        content_sequence = [float(x) for x in content_sequence]
        content_tensor = torch.tensor(content_sequence).type(torch.LongTensor)
        return content_tensor

    def __getitem__(self, idx):
        title = self.dataframe.iloc[idx]["preprocessed_title"]
        content = self.dataframe.iloc[idx]["preprocessed_content"]
        label = [self.dataframe.iloc[idx]["label"]]
        title_tensor = self._get_title_tensor(title=title)
        content_tensor = self._get_content_tensor(content=content)
        target_tensor = torch.FloatTensor(label).type(torch.LongTensor)

        return (title_tensor, content_tensor, target_tensor)
