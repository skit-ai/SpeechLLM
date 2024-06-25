import torch
from torch import nn


def get_connector(name, audio_enc_dim, llm_dim, k):
    if name == 'linear-pool':
        return LinearPoolConnector(audio_enc_dim, llm_dim, k)
    elif name == 'linear':
        return LinearConnector(audio_enc_dim, llm_dim, k)
    elif name == 'cnn':
        return CNNConnector(audio_enc_dim, llm_dim, k)
    else:
        raise NotImplementedError

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)

    def forward(self, x):
        x = self.layer(x)
        x = x.transpose(1, 2) 
        x = self.pool(x)  
        x = x.transpose(1, 2)
        return x


class LinearPoolConnector(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super(LinearPoolConnector, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU())
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.linear2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim))

    def forward(self, x):
        # x: [B, T, d]
        x = self.linear1(x)  # x: [B, T, D]
        x = x.transpose(1, 2)  # x: [B, D, T]
        x = self.pool(x)  # x: [B, D, T']
        x = x.transpose(1, 2)  # x: [B, T', D]
        x = self.linear2(x)
        return x

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels//2, kernel_size=5,
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels//2, out_channels, kernel_size=5,
                      stride=k, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,
                      stride=1, padding=0),
        )

    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)



if __name__ == "__main__":
    model = CNNConnector(128, 256)
    x = torch.randn(4, 50, 128)
    z = model(x)
    print(z.shape)