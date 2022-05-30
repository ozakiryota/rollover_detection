import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Discriminator, self).__init__()

        z_fc_out_dim = 512
        fc1_in_dim = 64 * (img_size // 4) * (img_size // 4) + z_fc_out_dim

        self.x_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.z_fc = nn.Linear(z_dim, z_fc_out_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(fc1_in_dim, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x, z):
        x_outputs = self.x_conv(x)
        z_outputs = self.z_fc(z)

        x_outputs = x_outputs.view(x.size(0), -1)
        outputs = torch.cat([x_outputs, z_outputs], dim=1)

        outputs = self.fc1(outputs)
        feature = outputs

        outputs = self.fc2(outputs)

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
    print("dis_outputs.size() =", dis_outputs.size())
    print(nn.Sigmoid()(dis_outputs))

if __name__ == '__main__':
    test()