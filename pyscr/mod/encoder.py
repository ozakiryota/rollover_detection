import torch.nn as nn

import sys
sys.path.append('../')
from mod.weights_initializer import initWeights

class Encoder(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Encoder, self).__init__()

        fc_in_dim = 4 * img_size * (img_size // 8) * (img_size // 8)

        self.conv = nn.Sequential(
            nn.Conv2d(3, img_size // 2, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(img_size // 2, img_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(img_size),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(img_size, 2 * img_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * img_size),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(2 * img_size, 4 * img_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4 * img_size),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, fc_in_dim // 8),
            nn.ReLU(inplace=True),

            nn.Linear(fc_in_dim // 8, fc_in_dim // 16),
            nn.ReLU(inplace=True),

            nn.Linear(fc_in_dim // 16, z_dim)
        )

        self.apply(initWeights)

    def forward(self, x):
        outputs = self.conv(x)
        outputs = outputs.view(x.size(0), -1)
        outputs = self.fc(outputs)
        return outputs


def test():
    import torch

    ## data
    batch_size = 10
    img_size = 112
    inputs = torch.randn(batch_size, 3, img_size, img_size)
    ## encode
    z_dim = 20
    enc_net = Encoder(z_dim, img_size)
    z = enc_net(inputs)
    ## debug
    print(enc_net)
    print("z.size() =", z.size())
    print("z[0] =", z[0])

if __name__ == '__main__':
    test()