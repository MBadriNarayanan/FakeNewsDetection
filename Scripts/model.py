import math
import torch
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Embedding,
    Flatten,
    Linear,
    Module,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class PositionalEncoding(Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class FakeNewsTransformer(Module):
    def __init__(
        self,
        title_vocab_size,
        content_vocab_size,
        encoder_dim,
        encoder_ffunits,
        encoder_layers,
        encoder_heads,
        dropout,
        activation,
        title_sequence_size,
        content_sequence_size,
        hidden_units_1,
        hidden_units_2,
        hidden_units_3,
        information_vector_dimension,
        device,
    ):
        super(FakeNewsTransformer, self).__init__()

        self.title_vocab_size = title_vocab_size
        self.content_vocab_size = content_vocab_size
        self.encoder_dim = encoder_dim
        self.encoder_ffunits = encoder_ffunits
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads
        self.dropout = dropout
        self.activation = activation
        self.title_sequence_size = title_sequence_size
        self.content_sequence_size = content_sequence_size
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.information_vector_dimension = information_vector_dimension
        self.device = device
        self.number_of_classes = 3

        self.TitleEmbedding = Embedding(self.title_vocab_size, self.encoder_dim)
        self.ContentEmbedding = Embedding(self.content_vocab_size, self.encoder_dim)

        self.PositionEncoding = PositionalEncoding(
            self.encoder_dim, dropout=self.dropout
        ).to(self.device)

        self.TitleEncoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=self.encoder_heads,
                dim_feedforward=self.encoder_ffunits,
                dropout=self.dropout,
                activation=self.activation,
                device=self.device,
            ),
            self.encoder_layers,
        ).to(self.device)

        self.ContentEncoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=self.encoder_heads,
                dim_feedforward=self.encoder_ffunits,
                dropout=self.dropout,
                activation=self.activation,
                device=self.device,
            ),
            self.encoder_layers,
        ).to(self.device)

        self.TitleReductionLayer = Sequential(
            Conv1d(in_channels=20, out_channels=80, kernel_size=2, stride=2),
        )

        self.ContentReductionLayer = Sequential(
            Linear(300, 150),
        )

        self.Decoder = Sequential(
            Conv1d(in_channels=80, out_channels=160, kernel_size=3, stride=2),
            BatchNorm1d(num_features=160),
            Conv1d(in_channels=160, out_channels=240, kernel_size=4, stride=2),
            BatchNorm1d(num_features=240),
            Conv1d(in_channels=240, out_channels=360, kernel_size=5, stride=2),
            BatchNorm1d(num_features=360),
            Flatten(),
            Linear(12600, self.hidden_units_1),
            Linear(self.hidden_units_1, self.hidden_units_2),
            Linear(self.hidden_units_2, self.hidden_units_3),
            Linear(self.hidden_units_3, self.number_of_classes),
        )

    def forward(self, title, content):

        title = self.TitleEmbedding(title) * math.sqrt(self.encoder_dim)
        content = self.ContentEmbedding(content) * math.sqrt(self.encoder_dim)

        title = title.permute(1, 0, 2)
        content = content.permute(1, 0, 2)

        title = self.PositionEncoding(title)
        content = self.PositionEncoding(content)

        title = self.TitleEncoder(title)
        content = self.ContentEncoder(content)

        title = title.permute(1, 0, 2)
        content = content.permute(1, 0, 2)

        title = self.TitleReductionLayer(title)
        content = self.ContentReductionLayer(content)

        information_vector = torch.cat((title, content), 2)

        output = self.Decoder(information_vector)

        return output
