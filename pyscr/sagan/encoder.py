import torch.nn as nn

import sys
sys.path.append('../')
from mod.weights_initializer import initWeights
from sagan.self_attention import SelfAttention

class Encoder(nn.Module):
    def __init__(self, z_dim, img_size, conv_unit_ch=32):
        super(Encoder, self).__init__()

        num_conv = 4
        last_conv_kernel = img_size // (2 ** num_conv)

        self.conv = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(3, conv_unit_ch, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.BatchNorm2d(conv_unit_ch),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),
            
            nn.utils.spectral_norm(
                nn.Conv2d(conv_unit_ch, 2 * conv_unit_ch, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.BatchNorm2d(2 * conv_unit_ch),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),
            
            nn.utils.spectral_norm(
                nn.Conv2d(2 * conv_unit_ch, 4 * conv_unit_ch, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.BatchNorm2d(4 * conv_unit_ch),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),

            SelfAttention(4 * conv_unit_ch),

            nn.utils.spectral_norm(
                nn.Conv2d(4 * conv_unit_ch, 8 * conv_unit_ch, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.BatchNorm2d(8 * conv_unit_ch),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),

            SelfAttention(8 * conv_unit_ch),

            nn.Conv2d(8 * conv_unit_ch, 16 * conv_unit_ch, kernel_size=last_conv_kernel, stride=1, bias=False),
            nn.BatchNorm2d(16 * conv_unit_ch),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16 * conv_unit_ch, 16 * conv_unit_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16 * conv_unit_ch),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16 * conv_unit_ch, z_dim, kernel_size=1, stride=1, bias=False)
        )

        self.apply(initWeights)

    def forward(self, x):
        outputs = self.conv(x)
        outputs = outputs.view(x.size(0), -1)
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