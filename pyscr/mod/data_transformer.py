from PIL import ImageOps
import numpy as np
import random
import math

from torchvision import transforms

class DataTransformer():
    def __init__(self, resize, mean, std, min_rollover_angle_deg):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.img_transformer = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        self.min_rollover_angle_deg = min_rollover_angle_deg

    def __call__(self, img_pil, acc_numpy, phase='train'):
        ## augemntation
        if phase == 'train':
            ## mirror
            is_mirror = bool(random.getrandbits(1))
            if is_mirror:
                img_pil, acc_numpy = self.mirror(img_pil, acc_numpy)
        ## img
        img_tensor = self.img_transformer(img_pil)
        ## acc
        is_rollover = self.isRollover(acc_numpy)
        return img_tensor, is_rollover

    def mirror(self, img_pil, acc_numpy):
        ## image
        img_pil = ImageOps.mirror(img_pil)
        ## acc
        acc_numpy[1] = -acc_numpy[1]
        return img_pil, acc_numpy

    def isRollover(self, acc_numpy):
        angle_rad = self.getAngleBetweenVectors(np.array([0, 0, 1]), acc_numpy)
        angle_deg = angle_rad / math.pi * 180.0
        return angle_deg > self.min_rollover_angle_deg

    def getAngleBetweenVectors(self, v1, v2):
        return math.acos(np.dot(v1, v2) / np.linalg.norm(v1, ord=2) / np.linalg.norm(v2, ord=2))


def test():
    import os
    from PIL import Image
    import matplotlib.pyplot as plt

    from datalist_maker import makeDataList

    ## data
    dir_list = [os.environ['HOME'] + '/dataset/rollover_detection/airsim/sample']
    csv_name = 'imu_camera.csv'
    data = makeDataList(dir_list, csv_name)[0]
    img_pil = Image.open(data[3])
    acc = [float(num) for num in data[:3]]
    acc = np.array(acc)
    ## transform
    resize = 224
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    min_rollover_angle_deg = 50.0
    data_transformer = DataTransformer(resize, mean, std, min_rollover_angle_deg)
    img_trans, label = data_transformer(img_pil, acc)
    ## debug
    print("img_trans.size() =", img_trans.size())
    print("label =", label)
    img_trans_numpy = img_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
    img_trans_numpy = np.clip(img_trans_numpy, 0, 1)
    plt.subplot(2, 1, 1)
    plt.imshow(img_pil)
    plt.subplot(2, 1, 2)
    plt.imshow(img_trans_numpy)
    plt.show()

if __name__ == '__main__':
    test()