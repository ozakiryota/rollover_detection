import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import random

import torch

import sys
sys.path.append('../')
from mod.datalist_maker import makeDataList
from mod.data_transformer import DataTransformer
from mod.dataset import RolloverDataset
from dcgan.generator import Generator as DcganG
from dcgan.discriminator import Discriminator as DcganD
from dcgan.encoder import Encoder as DcganE
from sagan.generator import Generator as SaganG
from sagan.discriminator import Discriminator as SaganD
from sagan.encoder import Encoder as SaganE
from mod.anomaly_score_computer import computeAnomalyScore

class Evaluator:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = self.setArgument().parse_args()
        self.dataset = self.getDataset()
        self.dis_net, self.gen_net, self.enc_net = self.getNetwork()
    
    def setArgument(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--dataset_dirs', required=True)
        arg_parser.add_argument('--csv_name', default='imu_camera.csv')
        arg_parser.add_argument('--img_size', type=int, default=112)
        arg_parser.add_argument('--z_dim', type=int, default=100)
        arg_parser.add_argument('--model_name', default='dcgan')
        arg_parser.add_argument('--conv_unit_ch', type=int, default=32)
        arg_parser.add_argument('--min_rollover_angle_deg', type=float, default=50.0)
        arg_parser.add_argument('--load_weights_dir', default='../../weights')
        arg_parser.add_argument('--save_fig_dir', default='../../fig')
        arg_parser.add_argument('--flag_show_reconstracted_images', action='store_true')
        arg_parser.add_argument('--show_h', type=int, default=5)
        arg_parser.add_argument('--show_w', type=int, default=10)

        return arg_parser

    def getDataset(self):
        ## data list
        dataset_dir_list = self.args.dataset_dirs.split('+')
        data_list = makeDataList(dataset_dir_list, self.args.csv_name)
        ## data transformer
        mean = ([0.5, 0.5, 0.5])
        std = ([0.5, 0.5, 0.5])
        data_transformer = DataTransformer(self.args.img_size, mean, std, self.args.min_rollover_angle_deg)
        ## dataset
        dataset = RolloverDataset(data_list, data_transformer, 'eval')

        return dataset

    def getNetwork(self):
        if self.args.model_name == 'sagan':
            dis_net = SaganD(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            gen_net = SaganG(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            enc_net = SaganE(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
        else:
            self.args.model_name = 'dcgan'
            dis_net = DcganD(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            gen_net = DcganG(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            enc_net = DcganE(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)

        gen_weights_path = os.path.join(self.args.load_weights_dir, 'generator.pth')
        dis_weights_path = os.path.join(self.args.load_weights_dir, 'discriminator.pth')
        enc_weights_path = os.path.join(self.args.load_weights_dir, 'encoder.pth')
        if torch.cuda.is_available():
            loaded_gen_weights = torch.load(gen_weights_path)
            print("load [GPU -> GPU]:", gen_weights_path)
            loaded_dis_weights = torch.load(dis_weights_path)
            print("load [GPU -> GPU]:", dis_weights_path)
            loaded_enc_weights = torch.load(enc_weights_path)
            print("load [GPU -> GPU]:", enc_weights_path)
        else:
            loaded_gen_weights = torch.load(gen_weights_path, map_location={"cuda:0": "cpu"})
            print("load [GPU -> CPU]:", gen_weights_path)
            loaded_dis_weights = torch.load(dis_weights_path, map_location={"cuda:0": "cpu"})
            print("load [GPU -> CPU]:", dis_weights_path)
            loaded_enc_weights = torch.load(enc_weights_path, map_location={"cuda:0": "cpu"})
            print("load [GPU -> CPU]:", enc_weights_path)
        gen_net.load_state_dict(loaded_gen_weights)
        dis_net.load_state_dict(loaded_dis_weights)
        enc_net.load_state_dict(loaded_enc_weights)

        dis_net.to(self.device)
        gen_net.to(self.device)
        enc_net.to(self.device)

        dis_net.eval()
        gen_net.eval()
        enc_net.eval()

        return dis_net, gen_net, enc_net

    def evaluate(self):
        images_list = []
        label_list = []
        score_list = []

        for i in range(len(self.dataset)):
            real_image = self.dataset.__getitem__(i)[0].unsqueeze(0).to(self.device)
            label = self.dataset.__getitem__(i)[1]

            z = self.enc_net(real_image)
            reconstracted_image = self.gen_net(z)

            _, real_feature = self.dis_net(real_image, z)
            _, reconstracted_feature = self.dis_net(reconstracted_image, z)

            anomaly_score = computeAnomalyScore(real_image, reconstracted_image, real_feature, reconstracted_feature).item()
            print("anomaly_score =", anomaly_score)

            images_list.append([real_image.squeeze(0).cpu().detach().numpy(), reconstracted_image.squeeze(0).cpu().detach().numpy()])
            label_list.append(label)
            score_list.append(anomaly_score)

        ## save
        random_indicies = list(range(len(score_list)))
        random.shuffle(random_indicies)
        self.saveSortedImages(images_list, label_list, random_indicies, self.args.show_h, self.args.show_w,
            'random' + str(self.args.show_h * self.args.show_w) + '.png')
        sorted_indicies = np.argsort(score_list)
        self.saveSortedImages(images_list, label_list, sorted_indicies, self.args.show_h, self.args.show_w,
            'top' + str(self.args.show_h * self.args.show_w) + '_smallest_score.png')
        self.saveSortedImages(images_list, label_list, sorted_indicies[::-1], self.args.show_h, self.args.show_w,
            'top' + str(self.args.show_h * self.args.show_w) + '_largest_score.png')
        plt.show()

    def saveSortedImages(self, images_list, label_list, indicies, h, w, save_name):
        num_shown = h * w

        if self.args.flag_show_reconstracted_images:
            h = 2 * h

        scale = 1.5
        plt.figure(figsize=(scale * h, scale * w))

        for i, index in enumerate(indicies):
            subplot_index = i + 1
            if subplot_index > num_shown:
                break
            
            if self.args.flag_show_reconstracted_images:
                subplot_index = 2 * w * (i // w) + (i % w) + 1
                reconstracted_image_numpy = images_list[index][1]
                reconstracted_image_numpy = np.clip(reconstracted_image_numpy.transpose((1, 2, 0)), 0, 1)
                plt.subplot(h, w, subplot_index + w, xlabel="g(e(x" + str(i + 1) + "))")
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                plt.imshow(reconstracted_image_numpy)

            real_image_numpy = images_list[index][0]
            real_image_numpy = np.clip(real_image_numpy.transpose((1, 2, 0)), 0, 1)
            sub_title = "rollover" if label_list[index] else ""
            plt.subplot(h, w, subplot_index, xlabel="x" + str(i + 1), ylabel=sub_title)
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            plt.imshow(real_image_numpy)
        plt.tight_layout()
        os.makedirs(self.args.save_fig_dir, exist_ok=True)
        plt.savefig(os.path.join(self.args.save_fig_dir, save_name))


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.evaluate()