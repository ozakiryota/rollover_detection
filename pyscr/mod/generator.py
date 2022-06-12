import torch.nn as nn

import sys
sys.path.append('../')
from mod.weights_initializer import initWeights

class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Generator, self).__init__()

        num_deconv = 4
        first_deconv_kernel = img_size // (2 ** num_deconv)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, kernel_size=first_deconv_kernel, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
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