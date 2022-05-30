import time

import torch
import torch.nn as nn

def train(gen_net, dis_net, enc_net, dataloader, num_epochs):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("device =", device)

    lr_ge = 0.0001
    lr_d = 0.0001/4
    beta1, beta2 = 0.5, 0.999
    gen_optimizer = torch.optim.Adam(gen_net.parameters(), lr_ge, [beta1, beta2])
    enc_optimizer = torch.optim.Adam(enc_net.parameters(), lr_ge, [beta1, beta2])
    dis_optimizer = torch.optim.Adam(dis_net.parameters(), lr_d, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    z_dim = 20
    # batch_size_in_loop = 64

    gen_net.to(device)
    enc_net.to(device)
    dis_net.to(device)

    gen_net.train()
    enc_net.train()
    dis_net.train()

    # torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    logs = []

    for epoch in range(num_epochs):

        epoch_start_clock = time.time()
        epoch_gen_loss = 0.0
        epoch_enc_loss = 0.0
        epoch_dis_loss = 0.0

        print("-------------")
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        for real_images, _ in dataloader:
            batch_size_in_loop = real_images.size(0)

            ## avoid batch normalization error
            if batch_size_in_loop == 1:
                continue

            real_labels = torch.full((batch_size_in_loop,), 1.0).to(device)
            fake_labels = torch.full((batch_size_in_loop,), 0.0).to(device)

            real_images = real_images.to(device)

            ## --------------------
            ## discrimator training
            ## --------------------
            real_z_encoded = enc_net(real_images)
            dis_outputs_real, _ = dis_net(real_images, real_z_encoded)

            fake_z_random = torch.randn(batch_size_in_loop, z_dim).to(device)
            fake_images = gen_net(fake_z_random)
            dis_outputs_fake, _ = dis_net(fake_images, fake_z_random)

            dis_loss_real = criterion(dis_outputs_real.view(-1), real_labels)
            dis_loss_fake = criterion(dis_outputs_fake.view(-1), fake_labels)
            dis_loss = dis_loss_real + dis_loss_fake

            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            # --------------------
            # generator training
            # --------------------
            fake_z_random = torch.randn(batch_size_in_loop, z_dim).to(device)
            fake_images = gen_net(fake_z_random)
            dis_outputs_fake, _ = dis_net(fake_images, fake_z_random)

            gen_loss = criterion(dis_outputs_fake.view(-1), real_labels)

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # --------------------
            # encoder training
            # --------------------
            real_z_encoded = enc_net(real_images)
            dis_outputs_real, _ = dis_net(real_images, real_z_encoded)

            enc_loss = criterion(dis_outputs_real.view(-1), fake_labels)

            enc_optimizer.zero_grad()
            enc_loss.backward()
            enc_optimizer.step()

            # --------------------
            # record
            # --------------------
            epoch_dis_loss += dis_loss.item()
            epoch_gen_loss += gen_loss.item()
            epoch_enc_loss += enc_loss.item()

        # epochのphaseごとのlossと正解率
        epoch_end_clock = time.time()
        print("Epoch_D_Loss:{:.4f} | Epoch_G_Loss:{:.4f} | Epoch_E_Loss:{:.4f}".format(epoch_dis_loss/batch_size, epoch_gen_loss/batch_size, epoch_enc_loss/batch_size))
        print("timer: {:.4f} sec.".format(epoch_end_clock - epoch_start_clock))

    return gen_net, dis_net, enc_net

import os

import sys
sys.path.append('../')
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
    csv_name = 'imu_camera.csv'
    data_list = makeDataList(dir_list, csv_name)
    ## transformer
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    min_rollover_angle_deg = 50.0
    data_transformer = DataTransformer(img_size, mean, std, min_rollover_angle_deg)
    ## dataset
    dataset = RolloverDataset(data_list, data_transformer, 'train')
    ## dataloader
    batch_size = 10
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ## network
    gen_net = Generator(z_dim, img_size)
    dis_net = Discriminator(z_dim, img_size)
    enc_net = Encoder(z_dim, img_size)
    ## train
    num_epochs = 10
    trained_gen_net, trained_dis_net, trained_enc_net = train(gen_net, dis_net, enc_net, dataloader, num_epochs)