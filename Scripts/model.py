import math
import torch
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Embedding,
    Flatten,
    GELU,
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
        embedding_flag,
        title_embedding_matrix,
        content_embedding_matrix,
        conv_output_units,
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
        self.embedding_flag = embedding_flag
        self.title_embedding_matrix = title_embedding_matrix
        self.content_embedding_matrix = content_embedding_matrix
        self.conv_output_units = conv_output_units
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.information_vector_dimension = information_vector_dimension
        self.device = device
        self.number_of_classes = 3

        if self.embedding_flag:
            self.TitleEmbedding = Embedding.from_pretrained(
                torch.FloatTensor(self.title_embedding_matrix), freeze=False
            )
            self.ContentEmbedding = Embedding.from_pretrained(
                torch.FloatTensor(self.content_embedding_matrix), freeze=False
            )

        else:
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
            Conv1d(
                in_channels=self.title_sequence_size,
                out_channels=self.content_sequence_size,
                kernel_size=2,
                stride=2,
            ),
            GELU(),
        )

        self.ContentReductionLayer = Sequential(
            Linear(self.encoder_dim, int(self.information_vector_dimension / 2)),
            GELU(),
        )

        self.Decoder = Sequential(
            Conv1d(
                in_channels=self.content_sequence_size,
                out_channels=self.content_sequence_size * 2,
                kernel_size=3,
                stride=2,
            ),
            BatchNorm1d(num_features=self.content_sequence_size * 2),
            GELU(),
            Conv1d(
                in_channels=self.content_sequence_size * 2,
                out_channels=self.content_sequence_size * 3,
                kernel_size=4,
                stride=2,
            ),
            BatchNorm1d(num_features=self.content_sequence_size * 3),
            GELU(),
            Conv1d(
                in_channels=self.content_sequence_size * 3,
                out_channels=self.content_sequence_size * 4,
                kernel_size=5,
                stride=2,
            ),
            BatchNorm1d(num_features=self.content_sequence_size * 4),
            GELU(),
            Flatten(),
            Linear(
                self.content_sequence_size * self.conv_output_units * 4,
                self.hidden_units_1,
            ),
            GELU(),
            Linear(self.hidden_units_1, self.hidden_units_2),
            GELU(),
            Linear(self.hidden_units_2, self.hidden_units_3),
            GELU(),
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
