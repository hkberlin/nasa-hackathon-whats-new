import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int) -> None:
        super(Seq2Seq, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dnn = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [bs, seq_len, input_dim]
        output, hn = self.rnn(seq)
        output = self.dnn(output)
        return output


class Seq2Val(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int) -> None:
        super(Seq2Val, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dnn = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [bs, seq_len, input_dim]
        output, hn = self.rnn(seq)
        output = output[:, -1, :]
        output = self.dnn(output)
        return output    
        
