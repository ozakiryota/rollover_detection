import torch.nn as nn

import sys
sys.path.append('../')
from mod.weights_initializer import initWeights

class Generator(nn.Module):
    def __init__(self, z_dim, img_size, conv_unit_ch=32):
        super(Generator, self).__init__()

        num_deconv = 4
        first_deconv_kernel = img_size // (2 ** num_deconv)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 8 * conv_unit_ch, kernel_size=first_deconv_kernel, stride=1, bias=False),
            nn.BatchNorm2d(8 * conv_unit_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8 * conv_unit_ch, 4 * conv_unit_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * conv_unit_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4 * conv_unit_ch, 2 * conv_unit_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * conv_unit_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2 * conv_unit_ch, conv_unit_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_unit_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(conv_unit_ch, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.apply(initWeights)

    def forward(self, z):
        outputs = z.view(z.size(0), -1, 1, 1)
        outputs = self.deconv(outputs)
        return outputs


def test():
    import numpy as np
    import matplotlib.pyplot as plt
    
    import torch
    
    ## network
    z_dim = 20
    img_size = 112
    gen_net = Generator(z_dim, img_size)
    gen_net.train()
    ## generate
    batch_size = 10
    input_z = torch.randn(batch_size, z_dim)
    fake_images = gen_net(input_z)
    ## debug
    print(gen_net)
    print("fake_images.size() =", fake_images.size())
    fake_img_numpy = np.clip(fake_images[0].detach().numpy().transpose((1, 2, 0)), 0, 1)
    plt.imshow(fake_img_numpy)
    plt.show()

if __name__ == '__main__':
    test()