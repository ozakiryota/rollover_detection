import torch.nn as nn

import sys
sys.path.append('../')
from mod.weights_initializer import initWeights

class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Generator, self).__init__()

        num_deconv = 4
        self.feature_size = img_size // (2 ** num_deconv)
        feature_ch = (2 ** (num_deconv - 1)) * img_size
        feature_dim = feature_ch * self.feature_size * self.feature_size

        self.fc = nn.Sequential(
            nn.Linear(z_dim, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),

            nn.Linear(feature_dim // 4, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(feature_dim // 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_ch, 4 * img_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * img_size),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4 * img_size, 2 * img_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * img_size),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2 * img_size, img_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(img_size),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(img_size, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.apply(initWeights)

    def forward(self, z):
        outputs = self.fc(z)
        outputs = outputs.view(z.size(0), -1, self.feature_size, self.feature_size)
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