from pathlib import Path

import torch
import torch.nn.functional as f
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Transformer):
    def __init__(self, ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5) -> None:
        super().__init__(d_model=ninp, nhead=nhead, num_encoder_layers=nlayers, dim_feedforward=nhid, dropout=dropout)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        nn.init.uniform(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform(self.decoder.weight, -initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_emb(src) * torch.sqrt(torch.tensor(self.ninp, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.encoder(src)
        output = self.decoder(output)
        return f.log_softmax(output, dim=-1)


class Tokenizer:
    def __init__(self) -> None:
        self.word2idx = {"<sos>": 0, "<eos>": 1}
        self.word_count = 2

    def add_word(self, word: str) -> None:
        if word not in self.word2idx:
            self.word_count += 1
            self.word2idx[word] = self.word_count - 1

    def tokenize(self, path: Path, length: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        if not path.exists():
            exp_msg = f"{path} does not exist."
            raise FileNotFoundError(exp_msg)

        with path.open(encoding="utf-8") as f:
            for line in f:
                words = ["<sos>", *line.split(), "<eos>"]
                for word in words:
                    self.add_word(word.lower())
        inputs = []
        output = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                words = [*line.split(), "<eos>"]
                prev_idx = 0
                cur_in = []
                cur_out = []
                for word in words:
                    token = self.word2idx[word.lower()]
                    if len(cur_in) > length - 1:
                        cur_in = cur_in[1:]
                    cur_in += [prev_idx]
                    if len(cur_in) < length:
                        cur_in = [self.word_count + 100] * (length - len(cur_in)) + cur_in
                    cur_out = [token]
                    prev_idx = token
                    inputs.append(cur_in)
                    output.append(cur_out)
                break
            inputs = torch.tensor(inputs, dtype=torch.long)
            output = torch.tensor(output, dtype=torch.long)
        return inputs, output


def main() -> None:
    file_path = "./docs/moby_dick.txt"
    path = Path(file_path)
    tokenizer = Tokenizer()
    results = tokenizer.tokenize(path=path)
    print(results)


if __name__ == "__main__":
    main()
