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
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_emb(src) * torch.sqrt(torch.tensor(self.ninp, dtype=torch.float32))
        print(src.size())
        src = src.transpose(0, 1)  # Convert to shape (seq_len, batch, features)
        src = self.pos_encoder(src)
        output = self.encoder(src)
        output = self.decoder(output)
        output = output.transpose(0, 1)  # Convert back to shape (batch, seq_len, features)
        result = f.log_softmax(output, dim=-1)
        return result


class Trainer(nn.Module):
    def __init__(self, ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = TransformerModel(ntoken, ninp, nhead, nhid, nlayers)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.NLLLoss()

    def train(self, src: torch.Tensor, target: torch.Tensor, epochs: int):
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(src)
            print(output.size())
            return None
            loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            self.optimizer.step()
            print(f"Loss: {loss.item()}")
        return self.model


class Tokenizer:
    def __init__(self) -> None:
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2}  # Added <pad> token
        self.word_count = 3  # Updated initial count

    def add_word(self, word: str) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = self.word_count
            self.word_count += 1

    def tokenize(self, path: Path, length: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        if not path.exists():
            exp_msg = f"{path} does not exist."
            raise FileNotFoundError(exp_msg)

        with path.open(encoding="utf-8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word.lower())
        inputs = []
        output = []
        cut_off = 20
        with path.open(encoding="utf-8") as f:
            for line in f:
                words = [*line.split(), "<eos>"]
                if len(words) < cut_off:
                    continue
                prev_idx = self.word2idx["<sos>"]  # Start with <sos> token
                cur_in = []
                cur_out = []
                for word in words:
                    token = self.word2idx[word.lower()]
                    if len(cur_in) > length - 1:
                        cur_in = cur_in[1:]
                    cur_in += [prev_idx]
                    if len(cur_in) < length:
                        cur_in = [self.word2idx["<pad>"]] * (length - len(cur_in)) + cur_in  # Use <pad> token
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
    x, y = tokenizer.tokenize(path=path)
    print(x.size(), y.size())
    trainer = Trainer(ntoken=tokenizer.word_count, ninp=x.size(1), nhead=1, nhid=200, nlayers=2)
    trainer.train(src=x, target=y, epochs=1)


if __name__ == "__main__":
    main()
