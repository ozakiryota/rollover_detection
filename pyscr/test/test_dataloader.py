import os
import numpy as np
import matplotlib.pyplot as plt

import torch

import sys
sys.path.append('../')
from mod.datalist_maker import makeDataList
# from mod.datalist_maker_without_csv import makeDataListWithoutCsv
from mod.data_transformer import DataTransformer
from mod.dataset import RolloverDataset

def showImages(images, h, w):
    num_shown = h * w
    for i, img in enumerate(images):
        if i + 1 > num_shown:
            break
        img = np.clip(img.cpu().detach().numpy().transpose((1, 2, 0)), 0, 1)
        plt.subplot(h, w, i + 1)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()

def test():
    ## data
    dir_list = [os.environ['HOME'] + '/dataset/rollover_detection/airsim/sample']
    csv_name = 'imu_camera.csv'
    data_list = makeDataList(dir_list, csv_name)
    # dir_list = [os.environ['HOME'] + '/dataset/img_align_celeba']
    # data_list = makeDataListWithoutCsv(dir_list)
    ## transformer
    resize = 224
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    data_transformer = DataTransformer(resize, mean, std)
    ## dataset
    dataset = RolloverDataset(data_list, data_transformer, 'train')
    ## dataloader
    batch_size = 10
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ## debug
    batch_iterator = iter(dataloader)
    inputs, labels = next(batch_iterator)
    print("inputs.size() = ", inputs.size())
    print("labels = ", labels)
    print("labels.size() = ", labels.size())
    showImages(inputs, 2, 5)

if __name__ == '__main__':
    test()