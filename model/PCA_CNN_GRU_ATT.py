import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        self.cnn_fc = nn.Linear(64, 128)
        self.gru = nn.GRU(input_size=3, hidden_size=32, num_layers=2, batch_first=True)
        self.gru_fc = nn.Linear(32, 128)
        self.Wa = nn.Parameter(torch.Tensor(128, 128))
        self.ba = nn.Parameter(torch.Tensor(128))
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        cnn_out = self.cnn(x.permute(0, 2, 1))
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = self.cnn_fc(cnn_out)
        gru_out, _ = self.gru(x)
        gru_out = self.gru_fc(
            gru_out[:, -1, :])
        Fi_cnn = torch.tanh(torch.mm(cnn_out, self.Wa.T) + self.ba)
        Fi_gru = torch.tanh(torch.mm(gru_out, self.Wa.T) + self.ba)
        Fi = Fi_cnn + Fi_gru
        A = torch.softmax(Fi, dim=1)
        FA = torch.sum(A.unsqueeze(2) * (cnn_out.unsqueeze(1) + gru_out.unsqueeze(1)), dim=1)
        x = self.dropout(torch.relu(self.fc1(FA)))
        x = self.fc2(x)
        return x
