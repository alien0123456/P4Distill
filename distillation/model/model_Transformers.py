

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        # Feature embedding
        self.len_vocab = args.len_vocab
        self.len_embedding_bits = args.len_embedding_bits
        self.ipd_vocab = args.ipd_vocab
        self.ipd_embedding_bits = args.ipd_embedding_bits
        self.embedding_vector_bits = args.embedding_vector_bits
        # Transformer parameters
        self.window_size = args.window_size
        self.rnn_in_pkts = args.rnn_in_pkts
        self.rnn_in_bits = args.rnn_in_pkts * args.embedding_vector_bits
        self.d_model = args.d_model
        self.rnn_hidden_bits = args.rnn_hidden_bits
        self.nhead = args.nhead
        self.num_layers = args.num_layers

        # Output layer
        self.labels_num = args.labels_num

        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits)
        self.fc = nn.Linear(self.len_embedding_bits + self.ipd_embedding_bits, self.embedding_vector_bits)
        self.input_projection = nn.Linear(self.rnn_in_bits, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.rnn_hidden_bits,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.out_layer = nn.Linear(self.d_model, self.labels_num)

    def forward(self, len_x, ipd_x):
        len_ebd = self.len_embedding(len_x)
        ipd_ebd = self.ipd_embedding(ipd_x)
        ebd_cat = torch.cat((len_ebd, ipd_ebd), dim=-1)
        batch_size = ebd_cat.shape[0]

        evs = self.fc(ebd_cat)
        rnn_in = evs.view(batch_size, -1, self.rnn_in_bits)
        rnn_in = self.input_projection(rnn_in)
        rnn_in = self.pos_encoder(rnn_in)
        
        transformer_out = self.transformer_encoder(rnn_in)
        final_output = transformer_out[:, -1, :]
        logits = self.out_layer(final_output)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x