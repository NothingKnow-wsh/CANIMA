from Conv_LSTM import *
from torch.nn import Sequential, Flatten, Module, Linear, Sigmoid, Softmax, ReLU, Tanh


class can_convlstm(Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, frame_size, batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first, bias, return_all_layers)
        self.fc_network = Sequential(
            Flatten(),
            Linear(frame_size * frame_size * 2, 128),
            Tanh(),
            Dropout(p=0.2),
            # Linear(128, 64),
            # Tanh(),
            # Dropout(p=0.2),
            Linear(128, 2),
            # Sigmoid()
            Softmax(dim=1)
        )

    def forward(self, x):
        _, lstm_out = self.convlstm(x)
        h = lstm_out[0][0]
        return self.fc_network(h)
