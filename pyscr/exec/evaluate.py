import torch

import sys
sys.path.append('../')
from mod.anomaly_score_computer import computeAnomalyScore

def evaluate(dataset, gen_net, dis_net, enc_net):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device =", device)

    gen_net.eval().to(device)
    dis_net.eval().to(device)
    enc_net.eval().to(device)

    for i in range(len(dataset)):
        real_image = dataset.__getitem__(i)[0].unsqueeze(0).to(device)

        z = enc_net(real_image)
        reconstracted_image = gen_net(z)

        _, real_feature = dis_net(real_image, z)
        _, reconstracted_feature = dis_net(reconstracted_image, z)

        anomaly_score = computeAnomalyScore(real_image, reconstracted_image, real_feature, reconstracted_feature)
        print("anomaly_score =", anomaly_score)

import os

from mod.datalist_maker import makeDataList
from mod.data_transformer import DataTransformer
from mod.dataset import RolloverDataset
from mod.generator import Generator
from mod.discriminator import Discriminator
from mod.encoder import Encoder

if __name__ == '__main__':
    ## parameters
    z_dim = 20
    img_size = 112
    ## data
    dir_list = [os.environ['HOME'] + '/dataset/rollover_detection/airsim/sample']
    # dir_list = [os.environ['HOME'] + '/dataset/rollover_detection/airsim/90deg']
    csv_name = 'imu_camera.csv'
    data_list = makeDataList(dir_list, csv_name)
    ## transformer
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    min_rollover_angle_deg = 50.0
    data_transformer = DataTransformer(img_size, mean, std, min_rollover_angle_deg)
    ## dataset
    dataset = RolloverDataset(data_list, data_transformer, 'eval')
    ## network
    gen_net = Generator(z_dim, img_size)
    dis_net = Discriminator(z_dim, img_size)
    enc_net = Encoder(z_dim, img_size)
    ## load
    weights_dir = '../../weights'
    gen_weights_path = os.path.join(weights_dir, 'generator.pth')
    dis_weights_path = os.path.join(weights_dir, 'discriminator.pth')
    enc_weights_path = os.path.join(weights_dir, 'encoder.pth')
    if torch.cuda.is_available():
        loaded_gen_weights = torch.load(gen_weights_path)
        print("Load [GPU -> GPU]:", gen_weights_path)
        loaded_dis_weights = torch.load(dis_weights_path)
        print("Load [GPU -> GPU]:", dis_weights_path)
        loaded_enc_weights = torch.load(enc_weights_path)
        print("Load [GPU -> GPU]:", enc_weights_path)
    else:
        loaded_gen_weights = torch.load(gen_weights_path, map_location={"cuda:0": "cpu"})
        print("Load [GPU -> CPU]:", gen_weights_path)
        loaded_dis_weights = torch.load(dis_weights_path, map_location={"cuda:0": "cpu"})
        print("Load [GPU -> CPU]:", dis_weights_path)
        loaded_enc_weights = torch.load(enc_weights_path, map_location={"cuda:0": "cpu"})
        print("Load [GPU -> CPU]:", enc_weights_path)
    gen_net.load_state_dict(loaded_gen_weights)
    dis_net.load_state_dict(loaded_dis_weights)
    enc_net.load_state_dict(loaded_enc_weights)
    ## evaluate
    evaluate(dataset, gen_net, dis_net, enc_net)