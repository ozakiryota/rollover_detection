import os
import numpy as np
import matplotlib.pyplot as plt

import torch

import sys
sys.path.append('../')
from dcgan.generator import Generator
# from sagan.generator import Generator

def test():
    ## device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## network
    z_dim = 20
    img_size = 112
    gen_net = Generator(z_dim, img_size)
    ## load
    weights_dir = '../../weights'
    gen_weights_path = os.path.join(weights_dir, 'generator.pth')
    if torch.cuda.is_available():
        loaded_weights = torch.load(gen_weights_path)
        print("Load [GPU -> GPU]:", gen_weights_path)
    else:
        loaded_weights = torch.load(gen_weights_path, map_location={"cuda:0": "cpu"})
        print("Load [GPU -> CPU]:", gen_weights_path)
    gen_net.load_state_dict(loaded_weights)
    gen_net.eval()
    gen_net.to(device)
    ## generate
    batch_size = 5
    input_z = torch.randn(batch_size, z_dim).to(device)
    fake_images = gen_net(input_z)
    ## show
    fake_images_numpy = fake_images.cpu().detach().numpy()
    for i, img in enumerate(fake_images_numpy):
        img = np.clip(img.transpose((1, 2, 0)), 0, 1)
        plt.subplot(1, batch_size, i+1)
        plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    test()