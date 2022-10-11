import os
import shutil
import socket
import datetime
import pickle
from dvae.model.dkf import DKF
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from .utils import myconf, get_logger, loss_ISD, loss_KLD, loss_MPJPE
from .model import build_DKF


class LearningAlgorithm():

    def __init__(self, params):
        # Load config parser
        self.params = params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_model(self):
            self.model = build_DKF(cfg=self.params, device=self.device)

    def init_optimizer(self):
        lr = self.params['l']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer

    def train(self):
        ############
        ### Init ###
        ############

        # Build model
        self.build_model()

        # Set module.training = True
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results

        saved_root = self.params['output_dir']
        z_dim = self.params['zdim']
        tag = 'DKF'
        filename = "{}_{}_z_dim={}".format('DKF', tag, z_dim)
        save_dir = os.path.join(saved_root, filename)
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = 1
        logger = get_logger(log_file, logger_type)

        # Init optimizer
        optimizer = self.init_optimizer()

        # Create data loader
        pkl = open(self.params['input'],'rb')
        data = pickle.load(pkl)
        data = np.array(data)
        data = torch.from_numpy(data)
        torch_dataset = Data.TensorDataset(data)
        train_dataloader = Data.DataLoader(dataset = torch_dataset, batch_size = self.params['b'], shuffle = False, num_workers = 0)
        val_dataloader = Data.DataLoader(dataset = torch_dataset, batch_size = self.params['b'], shuffle = False, num_workers = 0)
        train_num = 50000
        val_num = 50000
        ######################
        ### Batch Training ###
        ######################

        # Load training parameters
        epochs = self.params['n']
        early_stop_patience = 50
        save_frequency = 10
        beta = 0.1
        kl_warm = 0

        # Create python list for loss
        
        train_loss = np.zeros((epochs,))
        val_loss = np.zeros((epochs,))
        train_recon = np.zeros((epochs,))
        train_kl = np.zeros((epochs,))
        val_recon = np.zeros((epochs,))
        val_kl = np.zeros((epochs,))
        best_val_loss = np.inf
        cpt_patience = 0
        cur_best_epoch = epochs
        best_state_dict = self.model.state_dict()
        best_optim_dict = optimizer.state_dict()
        start_epoch = -1

        # Train with mini-batch SGD
        for epoch in range(start_epoch+1, epochs):
            print('epoch = ', epoch)
            start_time = datetime.datetime.now()

            # KL warm-up
            if epoch % 10 == 0 and kl_warm < 1:
                kl_warm = (epoch // 10) * 0.2 
                logger.info('KL warm-up, anneal coeff: {}'.format(kl_warm))

            # Batch training
            for _, batch_data in enumerate(train_dataloader):
                print('step = ', _)
                batch_data = (batch_data[0].unsqueeze(0)).to(self.device)
                batch_data = batch_data.permute(1, 0, 2)
                recon_batch_data, _, _, _, _, _ = self.model(batch_data)
                loss_function = torch.nn.MSELoss()
                loss_recon = loss_function(batch_data, recon_batch_data)
                seq_len, bs, _ = self.model.z_mean.shape
                loss_recon = loss_recon / (seq_len * bs)
                loss_kl = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                loss_kl = beta * loss_kl / (seq_len * bs)
                loss_tot = loss_recon + loss_kl
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss[epoch] += loss_tot.item() * bs
                train_recon[epoch] += loss_recon.item() * bs
                train_kl[epoch] += loss_kl.item() * bs

                print('loss kld = ', loss_kl)
                print('loss mse = ', loss_recon)
                print('loss total = ', loss_tot)
                
            # Validation
            for _, batch_data in enumerate(val_dataloader):
                batch_data = (batch_data[0].unsqueeze(0)).to(self.device)

                batch_data = batch_data.permute(1, 0, 2)
                recon_batch_data, _, _, _, _, _ = self.model(batch_data)
                loss_function = torch.nn.MSELoss()
                loss_recon = loss_function(batch_data, recon_batch_data)
                seq_len, bs, _ = self.model.z_mean.shape
                loss_recon = loss_recon / (seq_len * bs)
                loss_kl = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                loss_kl = kl_warm * beta * loss_kl / (seq_len * bs)
                loss_tot = loss_recon + loss_kl
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss[epoch] += loss_tot.item() * bs
                train_recon[epoch] += loss_recon.item() * bs
                train_kl[epoch] += loss_kl.item() * bs

                val_loss[epoch] += loss_tot.item() * bs
                val_recon[epoch] += loss_recon.item() * bs
                val_kl[epoch] += loss_kl.item() * bs

                print('loss total = ', loss_tot)

            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            train_recon[epoch] = train_recon[epoch] / train_num 
            train_kl[epoch] = train_kl[epoch]/ train_num
            val_recon[epoch] = val_recon[epoch] / val_num 
            val_kl[epoch] = val_kl[epoch] / val_num
            
            # Early stop patiance
            if val_loss[epoch] < best_val_loss or kl_warm <1:
                best_val_loss = val_loss[epoch]
                cpt_patience = 0
                best_state_dict = self.model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch
            else:
                cpt_patience += 1

            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
            logger.info('Train => tot: {:.6f} recon {:.6f} KL {:.6f} Val => tot: {:.6f} recon {:.6f} KL {:.6f}'.format(train_loss[epoch], train_recon[epoch], train_kl[epoch], val_loss[epoch], val_recon[epoch], val_kl[epoch]))

            # Stop traning if early-stop triggers
            if cpt_patience == early_stop_patience and kl_warm >= 1.0:
                logger.info('Early stop patience achieved')
                break

            # Save model parameters regularly
            if epoch % save_frequency == 0:
                loss_log = {'train_loss': train_loss[:cur_best_epoch+1],
                            'val_loss': val_loss[:cur_best_epoch+1],
                            'train_recon': train_recon[:cur_best_epoch+1],
                            'train_kl': train_kl[:cur_best_epoch+1], 
                            'val_recon': val_recon[:cur_best_epoch+1], 
                            'val_kl': val_kl[:cur_best_epoch+1]}
                save_file = os.path.join(save_dir, 'DKF' + '_checkpoint.pt')
                torch.save({'epoch': cur_best_epoch,
                            'best_val_loss': best_val_loss,
                            'cpt_patience': cpt_patience,
                            'model_state_dict': best_state_dict,
                            'optim_state_dict': best_optim_dict,
                            'loss_log': loss_log
                        }, save_file)
                logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))

        
        # Save the final weights of network with the best validation loss
        save_file = os.path.join(save_dir, 'DKF' + '_final_epoch' + str(cur_best_epoch) + '.pt')
        torch.save(best_state_dict, save_file)
        
        # Save the training loss and validation loss
        train_loss = train_loss[:epoch+1]
        val_loss = val_loss[:epoch+1]
        train_recon = train_recon[:epoch+1]
        train_kl = train_kl[:epoch+1]
        val_recon = val_recon[:epoch+1]
        val_kl = val_kl[:epoch+1]
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            pickle.dump([train_loss, val_loss, train_recon, train_kl, val_recon, val_kl], f)


        # Save the loss figure
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(fontsize=16, title='DKF', title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_{}.png'.format(tag))
        plt.savefig(fig_file)

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_recon, label='Training')
        plt.plot(val_recon, label='Validation')
        plt.legend(fontsize=16, title='{}: Recon. Loss'.format(DKF), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_recon_{}.png'.format(tag))
        plt.savefig(fig_file) 

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_kl, label='Training')
        plt.plot(val_kl, label='Validation')
        plt.legend(fontsize=16, title='{}: KL Divergence'.format(DKF), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_KLD_{}.png'.format(tag))
        plt.savefig(fig_file)


    

        