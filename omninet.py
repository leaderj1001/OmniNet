import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dims):
        super(MultiHeadAttention, self).__init__()
        self.in_dims = in_dims

        self.query = nn.Linear(in_dims, in_dims)
        self.key = nn.Linear(in_dims, in_dims)
        self.value = nn.Linear(in_dims, in_dims)

        self.softmax = nn.Softmax(dim=-1)

        self.out = nn.Linear(in_dims, in_dims)

    def forward(self, x, mask=None):
        n_batch, N, max_len = x.size(0), x.size(1), x.size(2)

        x = x.view(n_batch, max_len * N, -1)

        query = self.query(x)
        key = self.key(x)

        energy = torch.bmm(query, key.transpose(1, 2)) / (self.in_dims ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attn = self.softmax(energy)

        value = self.value(x)

        out = torch.bmm(attn, value)
        out = self.out(out).view(n_batch, -1, max_len, self.in_dims)

        return out


class FeedForward(nn.Module):
    def __init__(self, in_dims, ff_dims, dropout_rate=0.1):
        super(FeedForward, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dims, ff_dims),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(ff_dims, in_dims)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, in_dims, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(in_dims)
        self.norm1 = nn.LayerNorm(in_dims)

        self.ff = FeedForward(in_dims, in_dims * 4, dropout_rate)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x, mask=None):
        mha_out = self.mha(x, mask)
        mha_out = self.norm1(mha_out)
        mha_out += x

        ff_out = self.ff(mha_out)
        ff_out = self.norm2(ff_out)
        ff_out += mha_out

        return ff_out


class OmniNet(nn.Module):
    def __init__(self, in_dims, dims, N=12):
        super(OmniNet, self).__init__()
        self.dims = dims
        self.N = N
        self.embedding = nn.Linear(in_dims, dims)

        self.layers = nn.ModuleList([EncoderLayer(dims) for _ in range(N)])

        self.maxpool = nn.MaxPool1d(N)

    def forward(self, x, stack_x, mask=None):
        n_batch = x.size(0)
        x = self.embedding(x)
        stack_x = self.embedding(stack_x)

        for _ in range(self.N):
            stack_x = self.layers[_](stack_x, mask)
        o = self.maxpool(stack_x.view(n_batch, self.N, -1).transpose(1, 2)).view(n_batch, -1, self.dims)

        for _ in range(self.N):
            x = self.layers[_](x)

        out = x[:, -1] + o
        return out



def main():
    N = 12
    max_len = 32
    mask = torch.tril(torch.ones((N * max_len, N * max_len)))[(N - 1)::N].repeat((12, 1))

    x = torch.randn([2, 1, 32, 16])
    stack_x = torch.randn([2, 12, 32, 16])
    on = OmniNet(16, 512)
    print(on(x, stack_x, mask).size())


if __name__ == '__main__':
    main()
