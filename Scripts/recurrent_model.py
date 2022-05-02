from keras.models import Sequential
from keras.layers import Bidirectional, Embedding, Dense, Dropout, SimpleRNN, LSTM


class Model:
    def __init__(
        self,
        model_flag,
        bidirectional_flag,
        embedding_flag,
        sequence_length,
        vocab_size,
        embed_dim,
        embedding_matrix,
        dropout,
        recurrent_units,
        hidden_units,
    ):
        super(Model, self).__init__()
        self.model_flag = model_flag
        self.bidirectional_flag = bidirectional_flag
        self.embedding_flag = embedding_flag
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding_matrix = embedding_matrix
        self.dropout = dropout
        self.recurrent_units = recurrent_units
        self.hidden_units = hidden_units
        self.number_of_classes = 3

    def create_model(self):

        model = Sequential()

        if self.embedding_flag:

            self.embedding_matrix = [self.embedding_matrix]

            model.add(
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embed_dim,
                    input_length=self.sequence_length,
                    weights=self.embedding_matrix,
                )
            )

        else:

            model.add(
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embed_dim,
                    input_length=self.sequence_length,
                )
            )

        model.add(Dropout(self.dropout))

        if self.model_flag == "rnn":

            if self.bidirectional_flag:

                model.add(
                    Bidirectional(
                        SimpleRNN(units=self.recurrent_units, dropout=self.dropout)
                    )
                )
            else:

                model.add(SimpleRNN(units=self.recurrent_units, dropout=self.dropout))
        else:

            if self.bidirectional_flag:

                model.add(
                    Bidirectional(
                        LSTM(units=self.recurrent_units, dropout=self.dropout)
                    )
                )

            else:

                model.add(LSTM(units=self.recurrent_units, dropout=self.dropout))

        model.add(Dense(self.hidden_units, activation="relu"))
        model.add(Dense(self.number_of_classes, activation="softmax"))

        return model
