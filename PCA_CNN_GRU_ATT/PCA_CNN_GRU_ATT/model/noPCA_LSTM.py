import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=32, num_layers=2, batch_first=True)
        self.lstm_fc = nn.Linear(32, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = self.lstm_fc(lstm_out[:, -1, :])
        x = self.dropout(torch.relu(self.fc1(lstm_out)))
        x = self.fc2(x)
        return x
