import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        # Feature embedding
        self.len_vocab = args.len_vocab
        self.len_embedding_bits = args.len_embedding_bits
        self.ipd_vocab = args.ipd_vocab
        self.ipd_embedding_bits = args.ipd_embedding_bits
        self.embedding_vector_bits = args.embedding_vector_bits
        # RNN cell
        self.window_size = args.window_size
        self.rnn_in_pkts = args.rnn_in_pkts
        self.rnn_in_bits = args.rnn_in_pkts * args.embedding_vector_bits
        self.rnn_hidden_bits = args.rnn_hidden_bits
        # Output layer
        self.labels_num = args.labels_num

        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits)
        self.fc = nn.Linear(self.len_embedding_bits + self.ipd_embedding_bits, self.embedding_vector_bits)

        self.bilstm = nn.LSTM(
            input_size=self.rnn_in_bits,
            hidden_size=self.rnn_hidden_bits,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out_layer = nn.Linear(self.rnn_hidden_bits * 2, self.labels_num)

    def forward(self, len_x, ipd_x):
        len_ebd = self.len_embedding(len_x)
        ipd_ebd = self.ipd_embedding(ipd_x)
        ebd_cat = torch.cat((len_ebd, ipd_ebd), dim=-1)
        batch_size = ebd_cat.shape[0]

        evs = self.fc(ebd_cat)
        rnn_in = evs.view(batch_size, -1, self.rnn_in_bits)

        lstm_out, _ = self.bilstm(rnn_in)  # lstm_out: [batch_size, seq_len, hidden_size * 2]
        final_hidden = lstm_out[:, -1, :]

        logits = self.out_layer(final_hidden)
        return logits

    def get_evs(self, len_x, ipd_x):
        len_ebd = self.len_embedding(len_x)
        ipd_ebd = self.ipd_embedding(ipd_x)
        ebd_cat = torch.cat((len_ebd, ipd_ebd), dim=-1)
        batch_size = ebd_cat.shape[0]

        evs = self.fc(ebd_cat)
        return evs