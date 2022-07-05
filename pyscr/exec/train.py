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
from dcgan.generator import Generator as DcganG
from dcgan.discriminator import Discriminator as DcganD
from dcgan.encoder import Encoder as DcganE
from sagan.generator import Generator as SaganG
from sagan.discriminator import Discriminator as SaganD
from sagan.encoder import Encoder as SaganE
from mod.anomaly_score_computer import computeAnomalyScore

class Trainer:
    def __init__(self):
        self.args = self.setArgument().parse_args()
        self.checkArgument()
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.dataloader = self.getDataLoader()
        self.dis_net, self.gen_net, self.enc_net = self.getNetwork()
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.dis_optimizer, self.gen_optimizer, self.enc_optimizer = self.getOptimizer()
        self.info_str = self.getInfoStr()
        self.tb_writer = self.getWriter()
    
    def setArgument(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--device', default='cuda:0')
        arg_parser.add_argument('--dataset_dirs', required=True)
        arg_parser.add_argument('--csv_name', default='imu_camera.csv')
        arg_parser.add_argument('--img_size', type=int, default=112)
        arg_parser.add_argument('--z_dim', type=int, default=100)
        arg_parser.add_argument('--model_name', default='dcgan')
        arg_parser.add_argument('--conv_unit_ch', type=int, default=32)
        arg_parser.add_argument('--batch_size', type=int, default=100)
        arg_parser.add_argument('--load_weights_dir')
        arg_parser.add_argument('--flag_use_multi_gpu', action='store_true')
        arg_parser.add_argument('--lr_dis', type=float, default=5e-5)
        arg_parser.add_argument('--lr_gen', type=float, default=1e-5)
        arg_parser.add_argument('--lr_enc', type=float, default=1e-5)
        arg_parser.add_argument('--num_epochs', type=int, default=100)
        arg_parser.add_argument('--flag_use_gauss_z', action='store_true')
        arg_parser.add_argument('--loss_type', default='bce')
        arg_parser.add_argument('--save_weights_step', type=int)
        arg_parser.add_argument('--save_weights_dir', default='../../weights')
        arg_parser.add_argument('--save_log_dir', default='../../log')
        arg_parser.add_argument('--save_fig_dir', default='../../fig')

        return arg_parser

    def checkArgument(self):
        device_list = ['cpu', 'cuda'] + ['cuda:' + str(i) for i in range(torch.cuda.device_count())]
        if self.args.device not in device_list:
            self.args.device = 'cuda:0'
        if self.args.model_name not in ['dcgan', 'sagan']:
            self.args.model_name = 'dcgan'
        if self.args.loss_type not in ['bce', 'hinge']:
            self.args.loss_type = 'bce'
        if self.args.save_weights_step is None:
            self.args.save_weights_step = self.args.num_epochs

    def getDataLoader(self):
        ## data list
        dataset_dir_list = self.args.dataset_dirs.split('+')
        data_list = makeDataList(dataset_dir_list, self.args.csv_name)
        ## data transformer
        mean = ([0.5, 0.5, 0.5])
        std = ([0.5, 0.5, 0.5])
        data_transformer = DataTransformer(self.args.img_size, mean, std)
        ## dataset
        dataset = RolloverDataset(data_list, data_transformer, 'train')
        ## dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        return dataloader

    def getNetwork(self):
        if self.args.model_name == 'sagan':
            dis_net = SaganD(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            gen_net = SaganG(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            enc_net = SaganE(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
        else:
            dis_net = DcganD(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            gen_net = DcganG(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)
            enc_net = DcganE(self.args.z_dim, self.args.img_size, self.args.conv_unit_ch)

        if self.args.load_weights_dir is not None:
            gen_weights_path = os.path.join(self.args.load_weights_dir, 'generator.pth')
            dis_weights_path = os.path.join(self.args.load_weights_dir, 'discriminator.pth')
            enc_weights_path = os.path.join(self.args.load_weights_dir, 'encoder.pth')
            if self.device == torch.device('cpu'):
                loaded_gen_weights = torch.load(gen_weights_path, map_location={"cuda:0": "cpu"})
                print("load [GPU -> CPU]:", gen_weights_path)
                loaded_dis_weights = torch.load(dis_weights_path, map_location={"cuda:0": "cpu"})
                print("load [GPU -> CPU]:", dis_weights_path)
                loaded_enc_weights = torch.load(enc_weights_path, map_location={"cuda:0": "cpu"})
                print("load [GPU -> CPU]:", enc_weights_path)
            else:
                loaded_gen_weights = torch.load(gen_weights_path)
                print("load [GPU -> GPU]:", gen_weights_path)
                loaded_dis_weights = torch.load(dis_weights_path)
                print("load [GPU -> GPU]:", dis_weights_path)
                loaded_enc_weights = torch.load(enc_weights_path)
                print("load [GPU -> GPU]:", enc_weights_path)
            gen_net.load_state_dict(loaded_gen_weights)
            dis_net.load_state_dict(loaded_dis_weights)
            enc_net.load_state_dict(loaded_enc_weights)

        dis_net.to(self.device)
        gen_net.to(self.device)
        enc_net.to(self.device)
        if self.args.flag_use_multi_gpu:
            dis_net = nn.DataParallel(dis_net)
            gen_net = nn.DataParallel(gen_net)
            enc_net = nn.DataParallel(enc_net)

        return dis_net, gen_net, enc_net

    def getOptimizer(self):
        beta1, beta2 = 0.5, 0.999
        dis_optimizer = torch.optim.Adam(self.dis_net.parameters(), self.args.lr_dis, [beta1, beta2])
        gen_optimizer = torch.optim.Adam(self.gen_net.parameters(), self.args.lr_gen, [beta1, beta2])
        enc_optimizer = torch.optim.Adam(self.enc_net.parameters(), self.args.lr_enc, [beta1, beta2])

        return dis_optimizer, gen_optimizer, enc_optimizer

    def getInfoStr(self):
        info_str = self.args.model_name \
            + str(self.args.loss_type) \
            + str(self.args.img_size) + 'pixel' \
            + str(self.args.z_dim) + 'randz' \
            + str(self.args.conv_unit_ch) + 'ch' \
            + str(self.args.lr_dis) + 'lrd' \
            + str(self.args.lr_gen) + 'lrg' \
            + str(self.args.lr_enc) + 'lre' \
            + str(len(self.dataloader.dataset)) + 'sample' \
            + str(self.args.batch_size) + 'batch' \
            + str(self.args.num_epochs) + 'epoch'
        if self.args.load_weights_dir is not None:
            insert_index = info_str.find('epoch')
            info_str = info_str[:insert_index] + '+' + info_str[insert_index:]
        if self.args.flag_use_gauss_z:
            insert_index = info_str.find('randz')
            info_str = info_str[:insert_index] + 'gauss' + info_str[insert_index + len('rand'):]
        info_str = info_str.replace('-', '').replace('.', '')

        print("self.device =", self.device)
        print("info_str =", info_str)

        return info_str

    def getWriter(self):
        save_log_dir = os.path.join(self.args.save_log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + self.info_str)
        tb_writer = SummaryWriter(logdir=save_log_dir)
        print("save_log_dir =", save_log_dir)

        return tb_writer

    def exec(self):
        # torch.backends.cudnn.benchmark = True
        
        loss_record = []
        score_record = []
        start_clock = time.time()

        for epoch in range(self.args.num_epochs):
            epoch_start_clock = time.time()

            dis_epoch_loss = 0.0
            gen_epoch_loss = 0.0
            enc_epoch_loss = 0.0
            epoch_score = 0.0

            print("-------------")
            print("epoch: {}/{}".format(epoch + 1, self.args.num_epochs))

            for real_images, _ in self.dataloader:
                batch_size_in_loop = real_images.size(0)

                ## avoid batch normalization error
                if batch_size_in_loop == 1:
                    continue

                real_images = real_images.to(self.device)

                dis_loss, gen_loss, enc_loss = self.train(real_images, batch_size_in_loop)
                score = self.eval(real_images)

                dis_epoch_loss += batch_size_in_loop * dis_loss.item()
                gen_epoch_loss += batch_size_in_loop * gen_loss.item()
                enc_epoch_loss += batch_size_in_loop * enc_loss.item()
                epoch_score += batch_size_in_loop * score.item()
            self.record(epoch, loss_record, dis_epoch_loss, gen_epoch_loss, enc_epoch_loss, score_record, score)
            print("epoch time: {:.1f} sec".format(time.time() - epoch_start_clock))
            print("total time: {:.1f} min".format((time.time() - start_clock) / 60))

            if (epoch + 1) % self.args.save_weights_step == 0 or (epoch + 1) == self.args.num_epochs:
                self.saveWeights(epoch + 1)
        print("-------------")
        ## save
        self.tb_writer.close()
        self.saveLossGraph(loss_record)
        self.saveScoreGraph(score_record)
        plt.show()

    def train(self, real_images, batch_size_in_loop):
        self.dis_net.train()
        self.gen_net.train()
        self.enc_net.train()

        real_labels = torch.full((batch_size_in_loop,), 1.0).to(self.device)
        fake_labels = torch.full((batch_size_in_loop,), 0.0).to(self.device)

        ## --------------------
        ## discrimator training
        ## --------------------
        real_z_encoded = self.enc_net(real_images)
        dis_outputs_real, _ = self.dis_net(real_images, real_z_encoded)

        if self.args.flag_use_gauss_z:
            fake_z_random = torch.randn(batch_size_in_loop, self.args.z_dim).to(self.device)
        else:
            fake_z_random = torch.FloatTensor(batch_size_in_loop, self.args.z_dim).uniform_(-1.0, 1.0).to(self.device)
        fake_images = self.gen_net(fake_z_random)
        dis_outputs_fake, _ = self.dis_net(fake_images, fake_z_random)

        if self.args.loss_type == 'hinge':
            dis_loss_real = nn.ReLU()(1.0 - dis_outputs_real).mean()
            dis_loss_fake = nn.ReLU()(1.0 + dis_outputs_fake).mean()
        else:
            dis_loss_real = self.bce_criterion(dis_outputs_real.view(-1), real_labels)
            dis_loss_fake = self.bce_criterion(dis_outputs_fake.view(-1), fake_labels)
        dis_loss = dis_loss_real + dis_loss_fake

        self.dis_optimizer.zero_grad()
        dis_loss.backward()
        self.dis_optimizer.step()

        # --------------------
        # generator training
        # --------------------
        if self.args.flag_use_gauss_z:
            fake_z_random = torch.randn(batch_size_in_loop, self.args.z_dim).to(self.device)
        else:
            fake_z_random = torch.FloatTensor(batch_size_in_loop, self.args.z_dim).uniform_(-1.0, 1.0).to(self.device)
        fake_images = self.gen_net(fake_z_random)
        dis_outputs_fake, _ = self.dis_net(fake_images, fake_z_random)

        if self.args.loss_type == 'hinge':
            gen_loss = -dis_outputs_fake.mean()
        else:
            gen_loss = self.bce_criterion(dis_outputs_fake.view(-1), real_labels)

        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        # --------------------
        # encoder training
        # --------------------
        real_z_encoded = self.enc_net(real_images)
        dis_outputs_real, _ = self.dis_net(real_images, real_z_encoded)

        if self.args.loss_type == 'hinge':
            enc_loss = dis_outputs_real.mean()
        else:
            enc_loss = self.bce_criterion(dis_outputs_real.view(-1), fake_labels)

        self.enc_optimizer.zero_grad()
        enc_loss.backward()
        self.enc_optimizer.step()
        
        return dis_loss, gen_loss, enc_loss

    def eval(self, real_images):
        self.dis_net.eval()
        self.gen_net.eval()
        self.enc_net.eval()

        with torch.set_grad_enabled(False):
            real_z_encoded = self.enc_net(real_images)
            reconstracted_images = self.gen_net(real_z_encoded)
            _, real_feature = self.dis_net(real_images, real_z_encoded)
            _, reconstracted_feature = self.dis_net(reconstracted_images, real_z_encoded)

            reconstruction_score = computeAnomalyScore(real_images, reconstracted_images, real_feature, reconstracted_feature, 0.0)

        return reconstruction_score

    def record(self, epoch, loss_record, dis_epoch_loss, gen_epoch_loss, enc_epoch_loss, score_record, score):
        num_data = len(self.dataloader.dataset)
        loss_record.append([dis_epoch_loss / num_data, gen_epoch_loss / num_data, enc_epoch_loss / num_data])
        score_record.append(score / num_data)
        self.tb_writer.add_scalars('loss', {'dis': loss_record[-1][0], 'gen': loss_record[-1][1], 'enc': loss_record[-1][2]}, epoch)
        self.tb_writer.add_scalars('score', {'reconstruction': score_record[-1]}, epoch)
        print("loss: dis {:.4f} | gen {:.4f} | enc {:.4f}".format(loss_record[-1][0], loss_record[-1][1], loss_record[-1][2]))
        print("score: {:.4f}".format(score_record[-1]))

    def saveWeights(self, epoch):
        save_weights_dir = os.path.join(self.args.save_weights_dir, self.info_str)
        insert_index = save_weights_dir.find('batch') + len('batch')
        save_weights_dir = save_weights_dir[:insert_index] + str(epoch) + save_weights_dir[insert_index + len(str(self.args.num_epochs)):]
        os.makedirs(save_weights_dir, exist_ok=True)
        save_dis_weights_path = os.path.join(save_weights_dir, 'discriminator.pth')
        save_gen_weights_path = os.path.join(save_weights_dir, 'generator.pth')
        save_enc_weights_path = os.path.join(save_weights_dir, 'encoder.pth')
        if self.args.flag_use_multi_gpu:
            torch.save(self.dis_net.module.state_dict(), save_dis_weights_path)
            torch.save(self.gen_net.module.state_dict(), save_gen_weights_path)
            torch.save(self.enc_net.module.state_dict(), save_enc_weights_path)
        else:
            torch.save(self.dis_net.state_dict(), save_dis_weights_path)
            torch.save(self.gen_net.state_dict(), save_gen_weights_path)
            torch.save(self.enc_net.state_dict(), save_enc_weights_path)
        print("save:", save_dis_weights_path)
        print("save:", save_gen_weights_path)
        print("save:", save_enc_weights_path)

    def saveLossGraph(self, loss_record):
        plt.figure()
        loss_record_trans = list(zip(*loss_record))
        plt.plot(range(len(loss_record)), loss_record_trans[0], label="Dis")
        plt.plot(range(len(loss_record)), loss_record_trans[1], label="Gen")
        plt.plot(range(len(loss_record)), loss_record_trans[2], label="Enc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("loss: dis=" + '{:.4f}'.format(loss_record[-1][0]) + ", gen=" + '{:.4f}'.format(loss_record[-1][1]) + ", enc=" + '{:.4f}'.format(loss_record[-1][2]))

        fig_save_path = os.path.join(self.args.save_fig_dir, self.info_str + '_loss.jpg')
        plt.savefig(fig_save_path)

    def saveScoreGraph(self, score_record):
        plt.figure()
        plt.plot(range(len(score_record)), score_record, label="Reconstruction")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("score=" + '{:.4f}'.format(score_record[-1]))

        fig_save_path = os.path.join(self.args.save_fig_dir, self.info_str + '_score.jpg')
        plt.savefig(fig_save_path)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.exec()