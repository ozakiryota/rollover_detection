import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(ch, ch // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(ch, ch // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(ch, ch, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)

        outputs = torch.bmm(value, attention.permute(0, 2, 1))
        outputs = outputs.view(x.size(0), x.size(1), x.size(2), x.size(3))
        outputs = x + self.gamma * outputs

        return outputs


def test():
    ## network
    ch = 32
    attention_net = SelfAttention(ch)
    attention_net.train()
    ## generate
    batch_size = 10
    img_size = 64
    inputs = torch.randn(batch_size, ch, img_size, img_size)
    outputs = attention_net(inputs)
    ## debug
    print(attention_net)
    print("outputs.size() =", outputs.size())

if __name__ == '__main__':
    test()