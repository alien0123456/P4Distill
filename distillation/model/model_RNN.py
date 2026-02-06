import torch
import torch.nn as nn

class StandardGRUCell(nn.Module):
    def __init__(self, rnn_in_bits, rnn_hidden_bits):
        super(StandardGRUCell, self).__init__()
        self.cell = nn.GRUCell(rnn_in_bits, rnn_hidden_bits)
        self.rnn_in_bits = rnn_in_bits
        self.rnn_hidden_bits = rnn_hidden_bits

    def forward(self, input, time_steps):
        init_hidden = torch.zeros(input.size(0), self.rnn_hidden_bits, device=input.device)
        outputs = [init_hidden]

        for time_step in range(time_steps):
            batch = input[:, time_step]
            hidden = outputs[-1]

            next_hidden = self.cell(batch, hidden)
            outputs.append(next_hidden)

        final_hidden = outputs[-1]
        return final_hidden


class StandardRNN(nn.Module):
    def __init__(self, args):
        super(StandardRNN, self).__init__()
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
        self.standard_gru = StandardGRUCell(self.rnn_in_bits, self.rnn_hidden_bits)
        self.out_layer = nn.Linear(self.rnn_hidden_bits, self.labels_num)

    def forward(self, len_x, ipd_x):
        len_ebd = self.len_embedding(len_x)
        ipd_ebd = self.ipd_embedding(ipd_x)
        ebd_cat = torch.cat((len_ebd, ipd_ebd), dim=-1)
        batch_size = ebd_cat.shape[0]

        evs = self.fc(ebd_cat)
        rnn_in = evs.view(batch_size, -1, self.rnn_in_bits)
        rnn_out = self.standard_gru(rnn_in, self.window_size)
        logits = self.out_layer(rnn_out)
        return logits

    def get_evs(self, len_x, ipd_x):
        len_ebd = self.len_embedding(len_x)
        ipd_ebd = self.ipd_embedding(ipd_x)
        ebd_cat = torch.cat((len_ebd, ipd_ebd), dim=-1)
        batch_size = ebd_cat.shape[0]

        evs = self.fc(ebd_cat)
        return evs