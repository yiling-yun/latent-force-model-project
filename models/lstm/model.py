import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SocialGNN(nn.Module):
    def __init__(self, input_size, output_size, spatial_dim=12, hidden_dim=6, n_layers=1, drop_prob=0.0, apply_sigmoid=True):
        super(SocialGNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.apply_sigmoid = apply_sigmoid

        self.Gspatial = nn.Linear(input_size, spatial_dim)
        self.Gtemporal = nn.LSTM(spatial_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.drop = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lengths=None):
        batch_size, time_step, input_size = x.shape
        x = self.Gspatial(x.reshape(-1, input_size))   # spatial embedding
        x = x.view(batch_size, time_step, -1)
        hidden = self.init_hidden(batch_size)

        if lengths is not None:
            # use pack_padded_sequence for variable-length handling
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.Gtemporal(packed, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            # take last valid timestep per sequence
            idx = (lengths - 1).clamp(min=0)
            feat = lstm_out[torch.arange(batch_size), idx]
        else:
            lstm_out, _ = self.Gtemporal(x, hidden)
            feat = lstm_out[:, -1]  # last timestep

        out = self.drop(feat)
        out = self.fc(out)
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out, feat

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        device = next(self.parameters()).device
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=6, n_layers=1, drop_prob=0.0, apply_sigmoid=True):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.apply_sigmoid = apply_sigmoid

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers,
                            dropout=drop_prob if n_layers > 1 else 0,
                            batch_first=True)

        self.drop = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

        self._init_weights()

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

            # Run through LSTM
            # output contains the packed hidden states for all steps
            # hidden contains the FINAL hidden state for each sequence in the batch
            output, (hidden, cell) = self.lstm(x)

            feat = hidden[-1]
        else:
            _, (hidden, cell) = self.lstm(x)
            feat = hidden[-1]

        out = self.drop(feat)
        out = self.fc(out)
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out, feat

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights (Orthogonal is often preferred for LSTM)
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param.data, 0)
                # Forget gate bias trick: set to 1 to help long-term dependencies
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
            elif 'fc.weight' in name:
                # Output layer
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')

