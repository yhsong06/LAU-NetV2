import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *
from dataset import *
from util import *
import matplotlib.pyplot as plt
from torchvision import transforms
from preprocessing import load_segmented_data, load_segmented_data_val
from sklearn.model_selection import train_test_split
import logging
import time
from torchinfo import summary
import random
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "training.log")),
        logging.StreamHandler()
    ]
)


parser = argparse.ArgumentParser(description="Speech enhancement",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--lr", default=0.0004, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--network", default="LAUNetV2", type=str, dest="network")
parser.add_argument("--nch", default=2, type=int, dest="nch")
args, _ = parser.parse_known_args()
args = parser.parse_args()

mode = args.mode
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
network = args.network
nch = args.nch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)

logging.info("mode: %s", mode)
logging.info("learning rate: %.4e", lr)
logging.info("batch size: %d", batch_size)
logging.info("number of epoch: %d", num_epoch)
logging.info("network: %s", network)
logging.info("ckpt dir: %s", ckpt_dir)
logging.info("log dir: %s", log_dir)
logging.info("device: %s", device)


acc_seg_data, mic_seg_data, noisy_seg_data = load_segmented_data(save_dir='./segmented_data')
acc_seg_data_val, mic_seg_data_val, noisy_seg_data_val = load_segmented_data_val(save_dir='./segmented_data')

acc_seg_data = (acc_seg_data - np.min(acc_seg_data)) / (np.max(acc_seg_data) - np.min(acc_seg_data))
mic_seg_data = (mic_seg_data - np.min(mic_seg_data)) / (np.max(mic_seg_data) - np.min(mic_seg_data))
noisy_seg_data = (noisy_seg_data - np.min(noisy_seg_data)) / (np.max(noisy_seg_data) - np.min(noisy_seg_data))

acc_seg_data_val = (acc_seg_data_val - np.min(acc_seg_data_val)) / (np.max(acc_seg_data_val) - np.min(acc_seg_data_val))
mic_seg_data_val = (mic_seg_data_val - np.min(mic_seg_data_val)) / (np.max(mic_seg_data_val) - np.min(mic_seg_data_val))
noisy_seg_data_val = (noisy_seg_data_val - np.min(noisy_seg_data_val)) / (np.max(noisy_seg_data_val) - np.min(noisy_seg_data_val))

harmonic_masks = np.load("./segmented_data/harmonic_masks.npy")
harmonic_masks_val = np.load("./segmented_data/harmonic_masks_val.npy")
vad_masks = np.load("./segmented_data/vad_masks.npy")
vad_masks_val = np.load("./segmented_data/vad_masks_val.npy")


combined_data = list(zip(acc_seg_data, mic_seg_data, noisy_seg_data, harmonic_masks, vad_masks))
train_data = combined_data
combined_val_data = list(zip(acc_seg_data_val, mic_seg_data_val, noisy_seg_data_val, harmonic_masks_val, vad_masks_val))
val_data = combined_val_data

acc_train, mic_train, noisy_train, harmonic_train, vad_train = zip(*train_data)
logging.info(f"Shape: {len(acc_train)}")
logging.info(f"Shape: {len(mic_train)}")
logging.info(f"Shape: {len(noisy_train)}")
logging.info(f"Shape: {len(harmonic_train)}")
logging.info(f"Shape: {len(vad_train)}")
acc_val, mic_val, noisy_val, harmonic_val, vad_val = zip(*val_data)
logging.info(f"Shape: {len(acc_val)}")
logging.info(f"Shape: {len(mic_val)}")
logging.info(f"Shape: {len(noisy_val)}")
logging.info(f"Shape: {len(harmonic_val)}")
logging.info(f"Shape: {len(vad_val)}")


acc_train = torch.tensor(np.array(acc_train), dtype=torch.float32)
mic_train = torch.tensor(np.array(mic_train), dtype=torch.float32)
noisy_train = torch.tensor(np.array(noisy_train), dtype=torch.float32)
harmonic_train = torch.tensor(np.array(harmonic_train), dtype=torch.float32)
vad_train = torch.tensor(np.array(vad_train), dtype=torch.float32)

acc_val = torch.tensor(np.array(acc_val), dtype=torch.float32)
mic_val = torch.tensor(np.array(mic_val), dtype=torch.float32)
noisy_val = torch.tensor(np.array(noisy_val), dtype=torch.float32)
harmonic_val = torch.tensor(np.array(harmonic_val), dtype=torch.float32)
vad_val = torch.tensor(np.array(vad_val), dtype=torch.float32)


train_dataset = SpectrogramDataset_4CH(acc_train, mic_train, noisy_train, harmonic_train, vad_train)
val_dataset = SpectrogramDataset_4CH(acc_val, mic_val, noisy_val, harmonic_val, vad_val)


loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
num_data_train = len(train_dataset)
num_batch_train = np.ceil(num_data_train / batch_size)

loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
num_data_val = len(val_dataset)
num_batch_val = np.ceil(num_data_val / batch_size)


net = LAUNetV2(nch=nch).to(device)
fn_loss = nn.MSELoss().to(device)
optim = torch.optim.AdamW(net.parameters(), lr=lr)

train_loss_per_epoch = []
val_loss_per_epoch = []
best_val_loss = float('inf')


st_epoch = 0
if mode == 'train':
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_mse = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            label = label.unsqueeze(1)
            input = data['input'].to(device)
            
            output = net(input)

            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_mse += [loss.item()]            
            logging.info("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_mse)))

        avg_train_loss = np.mean(loss_mse)
        train_loss_per_epoch.append(avg_train_loss)

        
        with torch.no_grad():
            net.eval()
            loss_mse_val = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                label = label.unsqueeze(1)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)
                loss_mse_val += [loss.item()]

                logging.info("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_mse_val)))

            avg_val_loss = np.mean(loss_mse_val)
            val_loss_per_epoch.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
            logging.info("Checkpoint saved at EPOCH %04d with lowest VAL LOSS: %.4f" % (epoch, best_val_loss))

    plt.figure()
    plt.plot(range(1, len(train_loss_per_epoch) + 1), train_loss_per_epoch, label='Train Loss')
    plt.plot(range(1, len(val_loss_per_epoch) + 1), val_loss_per_epoch, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(ckpt_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Loss plot saved to {plot_path}")