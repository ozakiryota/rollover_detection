import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Encoder, self).__init__()

        conv_out_ch = 128
        fc_in_dim = conv_out_ch * (img_size // 4) * (img_size // 4)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, conv_out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.fc = nn.Linear(fc_in_dim, z_dim)

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
    print("z.size() =", z.size())
    print("z[0] =", z[0])

if __name__ == '__main__':
    test()