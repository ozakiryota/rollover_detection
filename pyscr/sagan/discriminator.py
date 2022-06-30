import torch
import torch.nn as nn

import sys
sys.path.append('../')
from mod.weights_initializer import initWeights
from sagan.self_attention import SelfAttention

class Discriminator(nn.Module):
    def __init__(self, z_dim, img_size, conv_unit_ch=32):
        super(Discriminator, self).__init__()

        num_conv = 4
        last_x_conv_kernel = img_size // (2 ** num_conv)

        self.x_conv = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(3, conv_unit_ch, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),

            nn.utils.spectral_norm(
                nn.Conv2d(conv_unit_ch, 2 * conv_unit_ch, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(2 * conv_unit_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),

            nn.utils.spectral_norm(
                nn.Conv2d(2 * conv_unit_ch, 4 * conv_unit_ch, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(4 * conv_unit_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),

            SelfAttention(4 * conv_unit_ch),

            nn.utils.spectral_norm(
                nn.Conv2d(4 * conv_unit_ch, 8 * conv_unit_ch, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(8 * conv_unit_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),

            SelfAttention(8 * conv_unit_ch),

            nn.Conv2d(8 * conv_unit_ch, 8 * conv_unit_ch, kernel_size=last_x_conv_kernel, stride=1)
        )

        self.z_conv = nn.Sequential(
            nn.Conv2d(z_dim, 8 * conv_unit_ch, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.2)
        )

        self.xz_conv1 = nn.Sequential(
            nn.Conv2d(16 * conv_unit_ch, 16 * conv_unit_ch, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout2d(p=0.2)
        )

        self.xz_conv2 = nn.Sequential(
            nn.Conv2d(16 * conv_unit_ch, 1, kernel_size=1, stride=1)
            # nn.Sigmoid()
        )

        self.apply(initWeights)

    def forward(self, x, z):
        x_outputs = self.x_conv(x)
        z_outputs = self.z_conv(z.view(z.size(0), -1, 1, 1))

        outputs = torch.cat([x_outputs, z_outputs], dim=1)

        outputs = self.xz_conv1(outputs)
        feature = outputs.view(x.size(0), -1)

        outputs = self.xz_conv2(outputs)
        outputs = outputs.view(x.size(0), -1)

        return outputs, feature


def test():
    ## data
    batch_size = 10
    z_dim = 20
    img_size = 112
    inputs_z = torch.randn(batch_size, z_dim)
    inputs_x = torch.randn(batch_size, 3, img_size, img_size)
    ## discriminate
    dis_net = Discriminator(z_dim, img_size)
    dis_outputs, _ = dis_net(inputs_x, inputs_z)
    ## debug
    print(dis_net)
    print("dis_outputs.size() =", dis_outputs.size())
    print("dis_outputs =", dis_outputs)

if __name__ == '__main__':
    test()