import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Generator, self).__init__()

        self.feature_size = img_size // 4

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, self.feature_size * self.feature_size * 128),
            nn.BatchNorm1d(self.feature_size * self.feature_size * 128),
            nn.ReLU(inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        outputs = self.fc(z)
        outputs = outputs.view(z.size(0), -1, self.feature_size, self.feature_size)
        outputs = self.deconv(outputs)
        return outputs


def test():
    import matplotlib.pyplot as plt
    
    import torch
    
    ## network
    z_dim = 20
    img_size = 112
    gen_net = Generator(z_dim, 112)
    gen_net.train()
    ## generate
    batch_size = 10
    input_z = torch.randn(batch_size, z_dim)
    fake_imgs = gen_net(input_z)
    ## debug
    print("fake_imgs.size() =", fake_imgs.size())
    fake_img_numpy = fake_imgs[0][0].detach().numpy()
    plt.imshow(fake_img_numpy)
    plt.show()

if __name__ == '__main__':
    test()