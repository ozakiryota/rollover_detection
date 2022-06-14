import argparse
import os
import time
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from mod.datalist_maker import makeDataList
from mod.data_transformer import DataTransformer
from mod.dataset import RolloverDataset
from mod.generator import Generator
from mod.discriminator import Discriminator
from mod.encoder import Encoder

class Trainer:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = self.setArgument()
        self.dataloader = self.getDataLoader()
        self.dis_net, self.gen_net, self.enc_net = self.getNetwork()
        self.dis_optimizer, self.gen_optimizer, self.enc_optimizer = self.getOptimizer()
        self.info_str = self.getInfoStr()
    
    def setArgument(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--dataset_dirs', required=True)
        arg_parser.add_argument('--csv_name', default='imu_camera.csv')
        arg_parser.add_argument('--img_size', type=int, default=112)
        arg_parser.add_argument('--z_dim', type=int, default=100)
        arg_parser.add_argument('--batch_size', type=int, default=100)
        arg_parser.add_argument('--load_weights_dir')
        arg_parser.add_argument('--lr_dis', type=float, default=5e-5)
        arg_parser.add_argument('--lr_gen', type=float, default=1e-5)
        arg_parser.add_argument('--lr_enc', type=float, default=1e-5)
        arg_parser.add_argument('--num_epochs', type=int, default=100)
        arg_parser.add_argument('--save_log_dir', default='../../log')
        arg_parser.add_argument('--save_weights_dir', default='../../weights')
        arg_parser.add_argument('--save_fig_dir', default='../../fig')

        return arg_parser.parse_args()

    def getDataLoader(self):
        ## data list
        dataset_dir_list = self.args.dataset_dirs.split('+')
        data_list = makeDataList(dataset_dir_list, self.args.csv_name)
        ## data transformer
        mean = ([0.5, 0.5, 0.5])
        std = ([0.5, 0.5, 0.5])
        min_rollover_angle_deg = 50.0
        data_transformer = DataTransformer(self.args.img_size, mean, std, min_rollover_angle_deg)
        ## dataset
        dataset = RolloverDataset(data_list, data_transformer, 'train')
        ## dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        return dataloader

    def getNetwork(self):
        dis_net = Discriminator(self.args.z_dim, self.args.img_size)
        gen_net = Generator(self.args.z_dim, self.args.img_size)
        enc_net = Encoder(self.args.z_dim, self.args.img_size)

        if self.args.load_weights_dir is not None:
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

        dis_net.train()
        gen_net.train()
        enc_net.train()

        return dis_net, gen_net, enc_net

    def getOptimizer(self):
        beta1, beta2 = 0.5, 0.999
        dis_optimizer = torch.optim.Adam(self.dis_net.parameters(), self.args.lr_dis, [beta1, beta2])
        gen_optimizer = torch.optim.Adam(self.gen_net.parameters(), self.args.lr_gen, [beta1, beta2])
        enc_optimizer = torch.optim.Adam(self.enc_net.parameters(), self.args.lr_enc, [beta1, beta2])

        return dis_optimizer, gen_optimizer, enc_optimizer

    def getInfoStr(self):
        info_str = str(len(self.dataloader.dataset)) + 'sample' \
            + str(self.args.img_size) + 'pixel' \
            + str(self.args.z_dim) + 'z' \
            + str(self.args.lr_dis) + 'lrd' \
            + str(self.args.lr_gen) + 'lrg' \
            + str(self.args.lr_enc) + 'lre' \
            + str(self.args.batch_size) + 'batch' \
            + str(self.args.num_epochs) + 'epoch'
        if self.args.load_weights_dir is not None:
            insert_index = info_str.find('epoch')
            info_str = info_str[:insert_index] + '+' + info_str[insert_index:]
        info_str = info_str.replace('-', '').replace('.', '')

        print("self.device =", self.device)
        print("info_str =", info_str)

        return info_str

    def train(self):
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

        # torch.backends.cudnn.benchmark = True
        
        ## buffer
        save_log_dir = os.path.join(self.args.save_log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + self.info_str)
        tb_writer = SummaryWriter(logdir=save_log_dir)
        loss_record = []
        start_clock = time.time()

        for epoch in range(self.args.num_epochs):

            epoch_start_clock = time.time()
            dis_epoch_loss = 0.0
            gen_epoch_loss = 0.0
            enc_epoch_loss = 0.0

            print("-------------")
            print("epoch: {}/{}".format(epoch + 1, self.args.num_epochs))

            for real_images, _ in self.dataloader:
                batch_size_in_loop = real_images.size(0)

                ## avoid batch normalization error
                if batch_size_in_loop == 1:
                    continue

                real_labels = torch.full((batch_size_in_loop,), 1.0).to(self.device)
                fake_labels = torch.full((batch_size_in_loop,), 0.0).to(self.device)

                real_images = real_images.to(self.device)

                ## --------------------
                ## discrimator training
                ## --------------------
                real_z_encoded = self.enc_net(real_images)
                dis_outputs_real, _ = self.dis_net(real_images, real_z_encoded)

                fake_z_random = torch.randn(batch_size_in_loop, self.args.z_dim).to(self.device)
                fake_images = self.gen_net(fake_z_random)
                dis_outputs_fake, _ = self.dis_net(fake_images, fake_z_random)

                dis_loss_real = criterion(dis_outputs_real.view(-1), real_labels)
                dis_loss_fake = criterion(dis_outputs_fake.view(-1), fake_labels)
                dis_loss = dis_loss_real + dis_loss_fake

                self.dis_optimizer.zero_grad()
                dis_loss.backward()
                self.dis_optimizer.step()

                # --------------------
                # generator training
                # --------------------
                fake_z_random = torch.randn(batch_size_in_loop, self.args.z_dim).to(self.device)
                fake_images = self.gen_net(fake_z_random)
                dis_outputs_fake, _ = self.dis_net(fake_images, fake_z_random)

                gen_loss = criterion(dis_outputs_fake.view(-1), real_labels)

                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()

                # --------------------
                # encoder training
                # --------------------
                real_z_encoded = self.enc_net(real_images)
                dis_outputs_real, _ = self.dis_net(real_images, real_z_encoded)

                enc_loss = criterion(dis_outputs_real.view(-1), fake_labels)

                self.enc_optimizer.zero_grad()
                enc_loss.backward()
                self.enc_optimizer.step()

                # --------------------
                # record
                # --------------------
                dis_epoch_loss += dis_loss.item()
                gen_epoch_loss += gen_loss.item()
                enc_epoch_loss += enc_loss.item()
            num_data = len(self.dataloader.dataset)
            loss_record.append([dis_epoch_loss / num_data, gen_epoch_loss / num_data, enc_epoch_loss / num_data])
            tb_writer.add_scalars("loss", {"dis": loss_record[-1][0], "gen": loss_record[-1][1], "enc": loss_record[-1][2]}, epoch)
            print("loss: dis {:.4f} | gen {:.4f} | enc {:.4f}".format(loss_record[-1][0], loss_record[-1][1], loss_record[-1][2]))
            print("epoch time: {:.1f} sec".format(time.time() - epoch_start_clock))
            print("total time: {:.1f} min".format((time.time() - start_clock) / 60))
        print("-------------")
        ## save
        tb_writer.close()
        self.saveWeights()
        self.saveLossGraph(loss_record)

    def saveLossGraph(self, loss_record):
        loss_record_trans = list(zip(*loss_record))
        plt.plot(range(len(loss_record)), loss_record_trans[0], label="Dis")
        plt.plot(range(len(loss_record)), loss_record_trans[1], label="Gen")
        plt.plot(range(len(loss_record)), loss_record_trans[2], label="Enc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("loss: dis=" + '{:.4f}'.format(loss_record[-1][0]) + ", gen=" + '{:.4f}'.format(loss_record[-1][1]) + ", enc=" + '{:.4f}'.format(loss_record[-1][2]))

        fig_save_path = os.path.join(self.args.save_fig_dir, self.info_str + '.jpg')
        plt.savefig(fig_save_path)
        plt.show()

    def saveWeights(self):
        save_weights_dir = os.path.join(self.args.save_weights_dir, self.info_str)
        os.makedirs(save_weights_dir, exist_ok=True)
        save_dis_weights_path = os.path.join(save_weights_dir, 'discriminator.pth')
        save_gen_weights_path = os.path.join(save_weights_dir, 'generator.pth')
        save_enc_weights_path = os.path.join(save_weights_dir, 'encoder.pth')
        torch.save(self.dis_net.state_dict(), save_dis_weights_path)
        torch.save(self.gen_net.state_dict(), save_gen_weights_path)
        torch.save(self.enc_net.state_dict(), save_enc_weights_path)
        print("save:", save_dis_weights_path)
        print("save:", save_gen_weights_path)
        print("save:", save_enc_weights_path)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()