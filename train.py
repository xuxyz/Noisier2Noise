import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from utils import make_exp_dir, denorm, get_transforms, tensor_to_numpy, LambdaLR
from models.DnCNN import DnCNN
from models.unet import UNet
from dataset import ImageNetGray

def get_model(model_type, device, input_channels=1):
    if model_type == 'dncnn':
        return DnCNN(channels=input_channels).to(device)
    elif model_type == 'unet':
        return UNet(input_channels=input_channels).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class TrainNr2N:
    def __init__(self, args):
        self.args = args
        # Device
        self.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
        # Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        # Params
        self.n_epochs = args.n_epochs
        self.start_epoch = args.start_epoch
        self.decay_epoch = args.decay_epoch
        self.lr = args.lr
        self.noise = args.noise
        # Loss
        self.criterion_mse = nn.MSELoss()
        # Data transforms
        transform = transforms.Compose(get_transforms(args))
        # Model
        self.model = get_model(args.model_type, self.device, input_channels=1)
        if args.load_model:
            load_path = f'./experiments/exp{args.load_exp_num}/checkpoints/{args.load_epoch}epochs.pth'
            self.model.load_state_dict(torch.load(load_path))
        # Datasets & loaders
        self.train_dataset = ImageNetGray(data_dir=args.data_root, noise=self.noise, train=True, transform=transform)
        self.test_dataset  = ImageNetGray(data_dir=args.data_root, noise=self.noise, train=False, transform=transform)
        # self.train_loader  = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        # self.val_loader    = DataLoader(self.test_dataset,  batch_size=args.batch_size, shuffle=False)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,      
            pin_memory=True,     
            persistent_workers=True  
        )
        
        self.val_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        
        # Optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_eps
        )
        lr_lambda = LambdaLR(self.n_epochs, self.start_epoch, self.decay_epoch).step
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=self.start_epoch-1)
        # Experiment dirs
        exp_info = make_exp_dir('./experiments/')
        self.exp_dir = exp_info['new_dir']
        self.exp_num = exp_info['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path    = os.path.join(self.exp_dir, 'results')
        # TensorBoard
        # self.summary = SummaryWriter(log_dir=f'runs/exp{self.exp_num}')
        # Metrics storage
        self.train_losses = []
        self.val_losses   = []
        self.psnr_list    = []
        self.ssim_list    = []
        self.metric_epochs = []

    def prepare(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_path,    exist_ok=True)
        # Save params
        with open(os.path.join(self.exp_dir, 'params.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(f"Training on {self.device}")
        self.prepare()
        for epoch in range(1, self.n_epochs + 1):
            # --- Training ---
            self.model.train()
            total_train_loss = 0.0
            for data in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                clean, noisy, noisier = data['clean'].to(self.device), data['noisy'].to(self.device), data['noisier'].to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model(noisier)
                loss = self.criterion_mse(prediction, noisy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) 
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train = total_train_loss / len(self.train_loader)
            self.train_losses.append(avg_train)
            # self.summary.add_scalar('train_loss', avg_train, epoch)
            # --- Validation Loss ---
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for data in self.val_loader:
                    clean, noisy, noisier = data['clean'].to(self.device), data['noisy'].to(self.device), data['noisier'].to(self.device)
                    pred = self.model(noisier)
                    total_val_loss += self.criterion_mse(pred, noisy).item()
            avg_val = total_val_loss / len(self.val_loader)
            self.val_losses.append(avg_val)
            # self.summary.add_scalar('val_loss', avg_val, epoch)
            # Step LR
            self.scheduler.step()


        
            # Checkpoints
            if epoch % 10 == 0 or epoch == self.n_epochs:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'{epoch}epochs.pth'))
            # --- PSNR & SSIM every 5 epochs ---
            if epoch % 5 == 0:
                noisy_psnr = output_psnr = prediction_psnr = 0.0
                noisy_ssim = output_ssim = prediction_ssim = 0.0
                num_samples = 10
                with torch.no_grad():
                    for idx in range(num_samples):
                        d = self.test_dataset[idx]
                        s_clean, s_noisy, s_noisier = d['clean'], d['noisy'], d['noisier']
                        s_noisy   = s_noisy.unsqueeze(0).to(self.device)
                        s_noisier = s_noisier.unsqueeze(0).to(self.device)
                        # out  = self.model(s_noisy)
                        pred = 2*self.model(s_noisier) - s_noisier
                        if self.args.normalize:
                            s_clean  = denorm(d['clean'], mean=self.args.mean, std=self.args.std)
                            s_noisy  = denorm(s_noisy,       mean=self.args.mean, std=self.args.std)
                            # out      = denorm(out,           mean=self.args.mean, std=self.args.std)
                            pred     = denorm(pred,          mean=self.args.mean, std=self.args.std)
                        sc_np = np.squeeze(tensor_to_numpy(s_clean))
                        sn_np = np.squeeze(tensor_to_numpy(s_noisy))
                        # out_np = np.squeeze(tensor_to_numpy(out))
                        pr_np = np.squeeze(tensor_to_numpy(pred))
                        noisy_psnr      += psnr(sc_np, sn_np, data_range=1)
                        # output_psnr     += psnr(sc_np, out_np, data_range=1)
                        prediction_psnr += psnr(sc_np, pr_np, data_range=1)
                        noisy_ssim      += ssim(sc_np, sn_np, data_range=1)
                        # output_ssim     += ssim(sc_np, out_np, data_range=1)
                        prediction_ssim += ssim(sc_np, pr_np, data_range=1)
                        if idx == 0:
                            clean_img = (255. * np.clip(sc_np, 0., 1.)).astype(np.uint8)
                            noisy_img = (255. * np.clip(sn_np, 0., 1.)).astype(np.uint8)
                            # out_img   = (255. * np.clip(out_np,0.,1.)).astype(np.uint8)
                            pr_img    = (255. * np.clip(pr_np,0.,1.)).astype(np.uint8)
                            cv2.imwrite(os.path.join(self.result_path, f'clean_{epoch}epochs.png'), clean_img)
                            cv2.imwrite(os.path.join(self.result_path, f'noisy_{epoch}epochs.png'), noisy_img)
                            # cv2.imwrite(os.path.join(self.result_path, f'output_{epoch}epochs.png'), out_img)
                            cv2.imwrite(os.path.join(self.result_path, f'prediction_{epoch}epochs.png'), pr_img)
                # output_psnr     /= num_samples
                # output_ssim     /= num_samples
                prediction_psnr     /= num_samples
                prediction_ssim     /= num_samples
                self.psnr_list.append(prediction_psnr)
                self.ssim_list.append(prediction_ssim)
                self.metric_epochs.append(epoch)
                print(f'Epoch {epoch} — Prediction PSNR: {prediction_psnr:.3f}, SSIM: {prediction_ssim:.3f}')






                plots_dir = os.path.join(self.exp_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)

                # Loss curves
                plt.figure()
                epochs = list(range(1, epoch+1))
                plt.plot(epochs, self.train_losses, label='Train Loss')
                plt.plot(epochs, self.val_losses, label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('MSE Loss')
                plt.legend()
                plt.title(f'Loss up to Epoch {epoch}')
                plt.savefig(os.path.join(plots_dir, f'exp{self.exp_num}_loss_up_to_{epoch}epochs.png'))
                plt.close()

                plt.figure()
                plt.plot(self.metric_epochs, self.psnr_list, label='PSNR')
                plt.xlabel('Epoch')
                plt.ylabel('PSNR')
                plt.legend()
                plt.title(f'PSNR up to Epoch {epoch}')
                plt.savefig(os.path.join(plots_dir, f'exp{self.exp_num}_psnr_up_to_{epoch}epochs.png'))
                plt.close()



        
                # self.summary.add_scalar('avg_output_psnr', output_psnr, epoch)
                # self.summary.add_scalar('avg_output_ssim', output_ssim, epoch)
        # --- After training: plots ---
        plots_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure()
        plt.plot(range(1, self.n_epochs+1), self.train_losses, label='Train Loss')
        plt.plot(range(1, self.n_epochs+1), self.val_losses,   label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title(f'Experiment {self.exp_num} Loss')
        plt.savefig(os.path.join(plots_dir, f'exp{self.exp_num}_loss.png'))
        plt.close()

        
        plt.figure()
        plt.plot(self.metric_epochs, self.psnr_list, label='PSNR')
        # plt.plot(self.metric_epochs, self.ssim_list, label='SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend()
        plt.title(f'Experiment {self.exp_num} PSNR')
        plt.savefig(os.path.join(plots_dir, f'exp{self.exp_num}_psnr.png'))
        plt.close()





        plt.figure()
        plt.plot(self.metric_epochs, self.ssim_list, label='SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.title(f'Experiment {self.exp_num} SSIM')
        plt.savefig(os.path.join(plots_dir, f'exp{self.exp_num}_ssim.png'))
        plt.close()
        # self.summary.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # define args: n_epochs, batch_size, gpu_num, seed, decay_epoch, lr, noise, data_root, model_type, beta1, beta2, weight_decay, adam_eps, mean, std, load_model, load_exp_num, load_epoch
    args = parser.parse_args()
    trainer = TrainNr2N(args)
    trainer.train()





# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriterx
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# import json
# import random
# from tqdm import tqdm

# from utils import *
# from models.DnCNN import DnCNN
# from models.unet import UNet
# from dataset import ImageNetGray

# def get_model(model_type, device, input_channels=1):
#     if model_type == 'dncnn':
#         return DnCNN(channels=input_channels).to(device)
#     elif model_type == 'unet':
#         return UNet(input_channels=input_channels).to(device)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")


# class TrainNr2N:
#     def __init__(self, args):
#         # Arguments
#         self.args = args

#         # Device
#         self.gpu_num = args.gpu_num
#         self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

#         # Random Seeds
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
#         np.random.seed(args.seed)

#         # Training Parameters
#         self.n_epochs = args.n_epochs
#         self.start_epoch = args.start_epoch
#         self.decay_epoch = args.decay_epoch
#         self.lr = args.lr
#         self.noise = args.noise
#         self.noise_type = self.noise.split('_')[0]
#         self.noise_intensity = float(self.noise.split('_')[1]) / 255.

#         # Loss
#         self.criterion_mse = nn.MSELoss()

#         # Transformation Parameters
#         self.mean = args.mean
#         self.std = args.std

#         # Transform
#         transform = transforms.Compose(get_transforms(args))

#         # Models
#         # self.model = DnCNN().to(self.device)
#         self.model = get_model(args.model_type, self.device, input_channels=1)  # NEW

        
#         if args.load_model:
#             load_path = './experiments/exp{}/checkpoints/{}epochs.pth'.format(args.load_exp_num, args.load_epoch)
#             self.model.load_state_dict(torch.load(load_path))

#         # Dataset
#         # self.train_dataset = ImageNetGray(noise=self.noise, train=True, transform=transform)
#         # self.test_dataset = ImageNetGray(noise=self.noise, train=False, transform=transform)

#         self.train_dataset = ImageNetGray(data_dir=args.data_root, noise=self.noise, train=True, transform=transform)
#         self.test_dataset = ImageNetGray(data_dir=args.data_root, noise=self.noise, train=False, transform=transform)
        
#         self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

#         # Optimizer
#         # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
#         # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-8)
        
#         self.optimizer = optim.Adam(
#                 self.model.parameters(),
#                 lr=self.lr,
#                 betas=(self.args.beta1, self.args.beta2),
#                 weight_decay=self.args.weight_decay,
#                 eps=self.args.adam_eps
#         )



#         # Scheduler
#         self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(self.n_epochs, self.start_epoch, self.decay_epoch).step)

#         # Directories
#         self.exp_dir = make_exp_dir('./experiments/')['new_dir']
#         self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
#         self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
#         self.result_path = os.path.join(self.exp_dir, 'results')

#         # Tensorboard
#         self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

#     def prepare(self):
#         # Save Paths
#         if not os.path.exists(self.checkpoint_dir):
#             os.makedirs(self.checkpoint_dir)

#         if not os.path.exists(self.result_path):
#             os.makedirs(self.result_path)

#         # Save Argument file
#         param_file = os.path.join(self.exp_dir, 'params.json')
#         with open(param_file, mode='w') as f:
#             json.dump(self.args.__dict__, f, indent=4)

#     def train(self):
#         print(self.device)
#         self.prepare()

#         for epoch in range(1, self.n_epochs + 1):
#             with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
#                 for batch, data in enumerate(tepoch):
#                     self.model.train()
#                     self.optimizer.zero_grad()

#                     clean, noisy, noisier = data['clean'], data['noisy'], data['noisier']
#                     clean, noisy, noisier = clean.to(self.device), noisy.to(self.device), noisier.to(self.device)

#                     prediction = self.model(noisier)
#                     loss = self.criterion_mse(prediction, noisy)
#                     loss.backward()
#                     self.optimizer.step()

#                     tepoch.set_postfix(rec_loss=loss.item())
#                     self.summary.add_scalar('loss', loss.item(), epoch)

#             self.scheduler.step()

#             # Checkpoints
#             if epoch % 10 == 0 or epoch == self.n_epochs:
#                 torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, '{}epochs.pth'.format(epoch)))

#             if epoch % 5 == 0:
#                 noisy_psnr, output_psnr, prediction_psnr = 0, 0, 0
#                 noisy_ssim, output_ssim, prediction_ssim = 0, 0, 0

#                 with torch.no_grad():
#                     self.model.eval()

#                     num_data = 10
#                     for index in range(num_data):
#                         data = self.test_dataset[index]
#                         sample_clean, sample_noisy, sample_noisier = data['clean'], data['noisy'], data['noisier']
#                         sample_noisy = torch.unsqueeze(sample_noisy, dim=0).to(self.device)
#                         sample_noisier = torch.unsqueeze(sample_noisier, dim=0).to(self.device)

#                         sample_output = self.model(sample_noisy)
#                         sample_prediction = 2*self.model(sample_noisier) - sample_noisier

#                         if self.args.normalize:
#                             sample_clean = denorm(sample_clean, mean=self.mean, std=self.std)
#                             sample_noisy = denorm(sample_noisy, mean=self.mean, std=self.std)
#                             sample_output = denorm(sample_output, mean=self.mean, std=self.std)
#                             sample_prediction = denorm(sample_prediction, mean=self.mean, std=self.std)

#                         sample_clean, sample_noisy = tensor_to_numpy(sample_clean), tensor_to_numpy(sample_noisy)
#                         sample_output, sample_prediction = tensor_to_numpy(sample_output), tensor_to_numpy(sample_prediction)

#                         sample_clean, sample_noisy = np.squeeze(sample_clean), np.squeeze(sample_noisy)
#                         sample_output, sample_prediction = np.squeeze(sample_output), np.squeeze(sample_prediction)

#                         # Calculate PSNR
#                         n_psnr = psnr(sample_clean, sample_noisy, data_range=1)
#                         o_psnr = psnr(sample_clean, sample_output, data_range=1)
#                         p_psnr = psnr(sample_clean, sample_prediction, data_range=1)
#                         # print('{}th image PSNR | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(index + 1, n_psnr, o_psnr, p_psnr))

#                         noisy_psnr += n_psnr / num_data
#                         output_psnr += o_psnr / num_data
#                         prediction_psnr += p_psnr / num_data

#                         # Calculate SSIM
#                         n_ssim = ssim(sample_clean, sample_noisy, data_range=1)
#                         o_ssim = ssim(sample_clean, sample_output, data_range=1)
#                         p_ssim = ssim(sample_clean, sample_prediction, data_range=1)
#                         # print('{}th image SSIM | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(index + 1, n_ssim, o_ssim, p_ssim))

#                         noisy_ssim += n_ssim / num_data
#                         output_ssim += o_ssim / num_data
#                         prediction_ssim += p_ssim / num_data

#                         # Save sample image
#                         sample_clean, sample_noisy = 255. * np.clip(sample_clean, 0., 1.), 255. * np.clip(sample_noisy, 0., 1.)
#                         sample_output, sample_prediction = 255. * np.clip(sample_output, 0., 1.), 255. * np.clip(sample_prediction, 0., 1.)

#                         if index == 0:
#                             cv2.imwrite(os.path.join(self.result_path, 'clean_{}epochs.png'.format(epoch)), sample_clean)
#                             cv2.imwrite(os.path.join(self.result_path, 'noisy_{}epochs.png'.format(epoch)), sample_noisy)
#                             cv2.imwrite(os.path.join(self.result_path, 'output_{}epochs.png'.format(epoch)), sample_output)
#                             cv2.imwrite(os.path.join(self.result_path, 'prediction_{}epochs.png'.format(epoch)), sample_prediction)

#                     # PSNR, SSIM
#                     print('Average PSNR | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(noisy_psnr, output_psnr, prediction_psnr))
#                     print('Average SSIM | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(noisy_ssim, output_ssim, prediction_ssim))
#                     self.summary.add_scalar('avg_output_psnr', output_psnr, epoch)
#                     self.summary.add_scalar('avg_output_ssim', output_ssim, epoch)
#                     self.summary.add_scalar('avg_prediction_psnr', prediction_psnr, epoch)
#                     self.summary.add_scalar('avg_prediction_ssim', prediction_ssim, epoch)

#         self.summary.close()









