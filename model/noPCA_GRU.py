import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size=6, hidden_size=32, num_layers=2, batch_first=True)
        self.gru_fc = nn.Linear(32, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.gru_fc(gru_out[:, -1, :])
        x = self.dropout(torch.relu(self.fc1(gru_out)))
        x = self.fc2(x)
        return x
