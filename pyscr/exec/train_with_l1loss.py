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

    def setArgument(self):
        arg_parser = super(TrainerWithL1Loss, self).setArgument()
        arg_parser.add_argument('--l1_loss_weight', type=float, default=0.1)

        return arg_parser

    def train(self):
        bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        l1_criterion = nn.L1Loss(reduction='mean')

        # torch.backends.cudnn.benchmark = True
        
        ## buffer
        save_log_dir = os.path.join(self.args.save_log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + self.info_str)
        tb_writer = SummaryWriter(logdir=save_log_dir)
        loss_record = []
        start_clock = time.time()

        for epoch in range(self.args.num_epochs):

            epoch_start_clock = time.time()
            dis_epoch_loss = 0.0
            gen_enc_epoch_loss = 0.0

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

                dis_loss_real = bce_criterion(dis_outputs_real.view(-1), real_labels)
                dis_loss_fake = bce_criterion(dis_outputs_fake.view(-1), fake_labels)
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

                bce_loss = bce_criterion(dis_outputs_fake.view(-1), real_labels) + bce_criterion(dis_outputs_real.view(-1), fake_labels)
                l1_loss = l1_criterion(real_images, reconstracted_images)
                gen_enc_loss = bce_loss + self.args.l1_loss_weight * l1_loss

                self.gen_optimizer.zero_grad()
                self.enc_optimizer.zero_grad()

                gen_enc_loss.backward()

                self.gen_optimizer.step()
                self.enc_optimizer.step()

                # --------------------
                # record
                # --------------------
                dis_epoch_loss += batch_size_in_loop * dis_loss.item()
                gen_enc_epoch_loss += batch_size_in_loop * gen_enc_loss.item()
            num_data = len(self.dataloader.dataset)
            loss_record.append([dis_epoch_loss / num_data, gen_enc_epoch_loss / num_data])
            tb_writer.add_scalars("loss", {"dis": loss_record[-1][0], "gen_enc": loss_record[-1][1]}, epoch)
            print("loss: dis {:.4f} | gen_enc {:.4f}".format(loss_record[-1][0], loss_record[-1][1]))
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
        plt.plot(range(len(loss_record)), loss_record_trans[1], label="GenEnc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("loss: dis=" + '{:.4f}'.format(loss_record[-1][0]) + ", gen_enc=" + '{:.4f}'.format(loss_record[-1][1]))

        fig_save_path = os.path.join(self.args.save_fig_dir, self.info_str + '.jpg')
        plt.savefig(fig_save_path)
        plt.show()

if __name__ == '__main__':
    trainer = TrainerWithL1Loss()
    trainer.train()