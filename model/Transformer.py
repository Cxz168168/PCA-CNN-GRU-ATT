import torch.nn.functional as F
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1, num_layers=4, nhead=4, dropout=0.1):
        super(Model, self).__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.output_linear = nn.Linear(hidden_size, output_size)
    def forward(self, src):
        src = self.input_linear(src)
        src = src.permute(1, 0, 2)
        memory = self.encoder(src)
        output = memory[-1, :, :]
        output = self.output_linear(output)
        return output