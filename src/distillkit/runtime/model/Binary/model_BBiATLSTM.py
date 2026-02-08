import torch
import torch.nn as nn
from ...utils.bnn_modules import Binary_a


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs: [batch_size, seq_len, hidden_size]
        attention_scores = torch.tanh(self.attention_weights(lstm_outputs))  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        weighted_output = torch.sum(attention_weights * lstm_outputs, dim=1)  # [batch_size, hidden_size]
        return weighted_output


class BiLSTMWithAttention(nn.Module):
    def __init__(self, args):
        super(BiLSTMWithAttention, self).__init__()
        # Feature embedding
        self.len_vocab = args.pkt_len_vocab_size
        self.len_embedding_bits = args.pkt_len_embed_bits
        self.ipd_vocab = args.ipd_vocab_size
        self.ipd_embedding_bits = args.ipd_embed_bits
        self.embedding_vector_bits = args.embed_dim_bits

        # BiLSTM parameters
        self.window_size = args.seq_window_size
        self.rnn_in_pkts = args.pkts_per_rnn_step
        self.rnn_in_bits = args.pkts_per_rnn_step * args.embed_dim_bits
        self.rnn_hidden_bits = args.rnn_hidden_state_bits

        # Output layer
        self.labels_num = args.num_classes

        self.activation_quantizer = Binary_a.apply

        # Embedding layers
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits)
        self.fc = nn.Linear(self.len_embedding_bits + self.ipd_embedding_bits, self.embedding_vector_bits)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.rnn_in_bits,
            hidden_size=self.rnn_hidden_bits,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanism (for bidirectional LSTM output)
        self.attention = Attention(2 * self.rnn_hidden_bits)

        # Output layer
        self.out_layer = nn.Linear(2 * self.rnn_hidden_bits, self.labels_num)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, len_x, ipd_x):
        len_ebd = self.len_embedding(len_x)
        len_ebd_bin = self.activation_quantizer(len_ebd)
        ipd_ebd = self.ipd_embedding(ipd_x)
        ipd_ebd_bin = self.activation_quantizer(ipd_ebd)
        ebd_bin_cat = torch.cat((len_ebd_bin, ipd_ebd_bin), dim=-1)
        batch_size = ebd_bin_cat.shape[0]

        evs = self.fc(ebd_bin_cat)
        evs_bin = self.activation_quantizer(evs)

        lstm_in_bin = evs_bin.view(batch_size, -1, self.rnn_in_bits)

        lstm_out, _ = self.lstm(lstm_in_bin)  # lstm_out: [batch_size, seq_len, 2 * hidden_size]

        final_hidden = self.attention(lstm_out)  # [batch_size, 2 * hidden_size]

        logits = self.out_layer(final_hidden)
        return logits

    def get_evs(self, len_x, ipd_x):
        len_ebd = self.len_embedding(len_x)
        len_ebd_bin = self.activation_quantizer(len_ebd)
        ipd_ebd = self.ipd_embedding(ipd_x)
        ipd_ebd_bin = self.activation_quantizer(ipd_ebd)
        ebd_bin_cat = torch.cat((len_ebd_bin, ipd_ebd_bin), dim=-1)
        batch_size = ebd_bin_cat.shape[0]

        evs = self.fc(ebd_bin_cat)
        evs_bin = self.activation_quantizer(evs)
        return evs_bin
