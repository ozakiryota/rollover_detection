import os

import torch

import sys
sys.path.append('../')
from mod.datalist_maker import makeDataList
from mod.data_transformer import DataTransformer
from mod.dataset import RolloverDataset

def test():
    ## data
    dir_list = [os.environ['HOME'] + '/dataset/rollover_detection/airsim/sample']
    csv_name = 'imu_camera.csv'
    data_list = makeDataList(dir_list, csv_name)
    ## transformer
    resize = 224
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    min_rollover_angle_deg = 50.0
    data_transformer = DataTransformer(resize, mean, std, min_rollover_angle_deg)
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

if __name__ == '__main__':
    test()