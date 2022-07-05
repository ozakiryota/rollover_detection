from PIL import Image
import numpy as np

import torch

class RolloverDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_transformer, phase):
        self.data_list = data_list
        self.data_transformer = data_transformer
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ## divide list
        img_path = self.data_list[index][3]
        acc_str_list = self.data_list[index][:3]
        acc_list = [float(num) for num in acc_str_list]
        ## to numpy
        img_pil = Image.open(img_path)
        acc_numpy = np.array(acc_list)
        ## tansform
        img_trans, label = self.data_transformer(img_pil, acc_numpy, phase=self.phase)
        return img_trans, label


def test():
    import os

    import datalist_maker
    from data_transformer import DataTransformer

    ## data
    dir_list = [os.environ['HOME'] + '/dataset/rollover_detection/airsim/sample']
    csv_name = 'imu_camera.csv'
    data_list = datalist_maker.makeDataList(dir_list, csv_name)
    ## transformer
    resize = 224
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    data_transformer = DataTransformer(resize, mean, std, min_rollover_angle_deg)
    ## dataset
    dataset = RolloverDataset(data_list, data_transformer, 'train')
    ## debug
    print("dataset.__len__() =", dataset.__len__())
    index = 0
    print("index", index, ": ", dataset.__getitem__(index)[0].size())
    print("index", index, ": ", dataset.__getitem__(index)[1])

if __name__ == '__main__':
    test()