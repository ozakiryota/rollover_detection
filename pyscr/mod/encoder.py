import torch.nn as nn

import sys
sys.path.append('../')
from mod.weights_initializer import initWeights

class Encoder(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Encoder, self).__init__()

        num_conv = 4
        last_conv_kernel = img_size // (2 ** num_conv)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(256, 512, kernel_size=last_conv_kernel, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, z_dim, kernel_size=1, stride=1, bias=False)
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