import torch.nn as nn

def initWeights(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.constant_(m.bias.data, 0)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif class_name.find('Linear') != -1:
        m.bias.data.fill_(0)


def test():
    from discriminator import Discriminator
    z_dim = 20
    img_size = 112
    dis_net = Discriminator(z_dim, img_size)
    dis_net.apply(initWeights)
    print(dis_net)

if __name__ == '__main__':
    test()