import argparse
import os
import time
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from train import Trainer

class TrainerWithL1Loss(Trainer):
    def __init__(self):
        super(TrainerWithL1Loss, self).__init__()
        self.l1_criterion = nn.L1Loss(reduction='mean')

    def setArgument(self):
        arg_parser = super(TrainerWithL1Loss, self).setArgument()
        arg_parser.add_argument('--l1_loss_weight', type=float, default=0.1)

        return arg_parser

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

        fake_z_random = torch.randn(batch_size_in_loop, self.args.z_dim).to(self.device)
        fake_images = self.gen_net(fake_z_random)
        dis_outputs_fake, _ = self.dis_net(fake_images, fake_z_random)

        dis_loss_real = self.bce_criterion(dis_outputs_real.view(-1), real_labels)
        dis_loss_fake = self.bce_criterion(dis_outputs_fake.view(-1), fake_labels)
        dis_loss = dis_loss_real + dis_loss_fake

        self.dis_optimizer.zero_grad()
        dis_loss.backward()
        self.dis_optimizer.step()

        # --------------------
        # generator & encoder training
        # --------------------
        fake_z_random = torch.randn(batch_size_in_loop, self.args.z_dim).to(self.device)
        fake_images = self.gen_net(fake_z_random)
        dis_outputs_fake, _ = self.dis_net(fake_images, fake_z_random)

        real_z_encoded = self.enc_net(real_images)
        dis_outputs_real, _ = self.dis_net(real_images, real_z_encoded)
        reconstracted_images = self.gen_net(real_z_encoded)

        bce_loss = self.bce_criterion(dis_outputs_fake.view(-1), real_labels) + self.bce_criterion(dis_outputs_real.view(-1), fake_labels)
        l1_loss = self.l1_criterion(real_images, reconstracted_images) + self.l1_criterion(fake_z_random, real_z_encoded)
        gen_enc_loss = bce_loss + self.args.l1_loss_weight * l1_loss

        self.gen_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()

        gen_enc_loss.backward()

        self.gen_optimizer.step()
        self.enc_optimizer.step()

        return dis_loss, gen_enc_loss, gen_enc_loss

    def record(self, epoch, loss_record, dis_epoch_loss, gen_epoch_loss, enc_epoch_loss, score_record, score):
        num_data = len(self.dataloader.dataset)
        loss_record.append([dis_epoch_loss / num_data, gen_epoch_loss / num_data])
        score_record.append(score / num_data)
        self.tb_writer.add_scalars('loss', {'dis': loss_record[-1][0], 'gen_enc': loss_record[-1][1]}, epoch)
        self.tb_writer.add_scalars('score', {'reconstruction': score_record[-1]}, epoch)
        print("loss: dis {:.4f} | gen_enc {:.4f}".format(loss_record[-1][0], loss_record[-1][1]))
        print("score: {:.4f}".format(score_record[-1]))

    def saveLossGraph(self, loss_record):
        plt.figure()
        loss_record_trans = list(zip(*loss_record))
        plt.plot(range(len(loss_record)), loss_record_trans[0], label="Dis")
        plt.plot(range(len(loss_record)), loss_record_trans[1], label="GenEnc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("loss: dis=" + '{:.4f}'.format(loss_record[-1][0]) + ", gen_enc=" + '{:.4f}'.format(loss_record[-1][1]))

        fig_save_path = os.path.join(self.args.save_fig_dir, self.info_str + '_loss.jpg')
        plt.savefig(fig_save_path)

if __name__ == '__main__':
    trainer = TrainerWithL1Loss()
    trainer.exec()