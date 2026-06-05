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
from preprocessing import load_segmented_data_test
from sklearn.model_selection import train_test_split
import logging
import time
from torchinfo import summary
import librosa
import soundfile as sf
from compute_metric import *
import pandas as pd
from pystoi import stoi 
from pesq import pesq   
import scipy.signal as signal
import xlsxwriter
from fvcore.nn import FlopCountAnalysis, parameter_count
import re


log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "test.log")),
        logging.StreamHandler()
    ]
)


parser = argparse.ArgumentParser(description="Speech enhancement test",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="test", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--lr", default=0.0004, type=float, dest="lr")
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=1, type=int, dest="num_epoch")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--network", default="LAUNetV2", type=str, dest="network")
parser.add_argument("--nch", default=2, type=int, dest="nch")
args, _ = parser.parse_known_args()
args = parser.parse_args()

mode = args.mode
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
ckpt_dir = args.ckpt_dir
network = args.network
nch = args.nch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info("mode: %s", mode)
logging.info("learning rate: %.4e", lr)
logging.info("batch size: %d", batch_size)
logging.info("number of epoch: %d", num_epoch)
logging.info("network: %s", network)
logging.info("ckpt dir: %s", ckpt_dir)
logging.info("device: %s", device)


sound_file_name, acc_seg_data, mic_seg_data, noisy_seg_data, acc_seg_data_phase, mic_seg_data_phase, noisy_seg_data_phase = load_segmented_data_test(save_dir='./segmented_data')

acc_seg_data = acc_seg_data.astype(np.float32)
mic_seg_data = mic_seg_data.astype(np.float32)
noisy_seg_data = noisy_seg_data.astype(np.float32)
acc_seg_data_phase = acc_seg_data_phase.astype(np.float32)
mic_seg_data_phase = mic_seg_data_phase.astype(np.float32)
noisy_seg_data_phase = noisy_seg_data_phase.astype(np.float32)

harmonic_masks = np.load("./segmented_data/harmonic_masks_test.npy")
vad_masks = np.load("./segmented_data/vad_masks_test.npy")

acc_min, acc_max = np.min(acc_seg_data), np.max(acc_seg_data)
mic_min, mic_max = np.min(mic_seg_data), np.max(mic_seg_data)
noisy_min, noisy_max = np.min(noisy_seg_data), np.max(noisy_seg_data)
acc_seg_data = (acc_seg_data - acc_min) / (acc_max - acc_min)
mic_seg_data = (mic_seg_data - mic_min) / (mic_max - mic_min)
noisy_seg_data = (noisy_seg_data - noisy_min) / (noisy_max - noisy_min)

logging.info(f"Shape: {acc_seg_data.shape}")
logging.info(f"Shape: {mic_seg_data.shape}")
logging.info(f"Shape: {noisy_seg_data.shape}")


acc_test = torch.tensor(np.array(acc_seg_data), dtype=torch.float32)
mic_test = torch.tensor(np.array(mic_seg_data), dtype=torch.float32)
noisy_test = torch.tensor(np.array(noisy_seg_data), dtype=torch.float32)
harmonic_test = torch.tensor(np.array(harmonic_masks), dtype=torch.float32)
vad_test = torch.tensor(np.array(vad_masks), dtype=torch.float32)


test_dataset = SpectrogramDataset_4CH(acc_test, mic_test, noisy_test, harmonic_test, vad_test)
loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
num_data_test = len(test_dataset)
num_batch_test = np.ceil(num_data_test / batch_size)


net = LAUNetV2(nch=nch).to(device)
fn_loss = nn.MSELoss().to(device)
optim = torch.optim.AdamW(net.parameters(), lr=lr)


fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

st_epoch = 0
sr = 8000

output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# TEST MODE
if mode == 'test':
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

        
    with torch.no_grad():
        net.eval()
        loss_mse = []
        inference_time =[]

        combined_sound_data = {}
        count = 0

        for batch, data in enumerate(loader_test, 1):

            label = data['label'].to(device)
            label = label.unsqueeze(1)
            input = data['input'].to(device)

            output = net(input)

            loss = fn_loss(output, label)
            loss_mse += [loss.item()]
            logging.info("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_mse)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(output)

            input_acc = input[..., 0].squeeze()
            input_noisy = input[..., 1].squeeze()
            label_rgb = label[0].squeeze()
            output_rgb = output[0].squeeze()

            input_acc = input_acc * (acc_max - acc_min) + acc_min
            input_noisy = input_noisy * (noisy_max - noisy_min) + noisy_min
            output_rgb = np.clip(output_rgb, 0, 1)
            output_rgb = output_rgb * (noisy_max - noisy_min) + noisy_min
            label_rgb = label_rgb * (mic_max - mic_min) + mic_min


            current_sound_name = sound_file_name[batch - 1]
            if current_sound_name not in combined_sound_data:
                combined_sound_data[current_sound_name] = {
					'input_acc': [],
					'input_noisy': [],
					'output': [],
					'label': [],
                    'acc_phase': [],
                    'mic_phase': [],
                    'noisy_phase': [],
                    'count': []
				}

            combined_sound_data[current_sound_name]['input_acc'].append(input_acc)  # Accumulated input
            combined_sound_data[current_sound_name]['input_noisy'].append(input_noisy)  # Noisy input
            combined_sound_data[current_sound_name]['output'].append(output_rgb)     # Model output
            combined_sound_data[current_sound_name]['label'].append(label_rgb)       # Ground truth label
            combined_sound_data[current_sound_name]['acc_phase'].append(acc_segment)       # acc_segment vs. noisy_segment vs. mic_segment
            combined_sound_data[current_sound_name]['mic_phase'].append(mic_segment)       # acc_segment vs. noisy_segment vs. mic_segment
            combined_sound_data[current_sound_name]['noisy_phase'].append(noisy_segment)       # acc_segment vs. noisy_segment vs. mic_segment
            combined_sound_data[current_sound_name]['count'].append(count)       

            count = count + 1

        results = []
        dataset_dir = r'D:\PUT_YOUR_DATASET_DIRECTORY'
        subject_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

        for sound_name, sound_data in combined_sound_data.items():

            combined_input_acc = np.concatenate(sound_data['input_acc'], axis=1)
            combined_input_noisy = np.concatenate(sound_data['input_noisy'], axis=1)
            combined_output = np.concatenate(sound_data['output'], axis=1)
            combined_label = np.concatenate(sound_data['label'], axis=1)
            combined_acc_phase = np.concatenate(sound_data['acc_phase'], axis=1)
            combined_mic_phase = np.concatenate(sound_data['mic_phase'], axis=1)
            combined_noisy_phase = np.concatenate(sound_data['noisy_phase'], axis=1)


            combined_input_acc_amplitude = librosa.db_to_amplitude(combined_input_acc, ref=1.0)
            combined_input_noisy_amplitude = librosa.db_to_amplitude(combined_input_noisy, ref=1.0)
            combined_output_amplitude = librosa.db_to_amplitude(combined_output, ref=1.0)
            combined_label_amplitude = librosa.db_to_amplitude(combined_label, ref=1.0)


            combined_input_acc_amplitude = np.vstack([np.zeros((1, combined_input_acc_amplitude.shape[1])), combined_input_acc_amplitude])
            combined_acc_phase = np.vstack([np.zeros((1, combined_acc_phase.shape[1])), combined_acc_phase])
            
            combined_input_noisy_amplitude = np.vstack([np.zeros((1, combined_input_noisy_amplitude.shape[1])), combined_input_noisy_amplitude])
            combined_noisy_phase = np.vstack([np.zeros((1, combined_noisy_phase.shape[1])), combined_noisy_phase])
            
            combined_output_amplitude = np.vstack([np.zeros((1, combined_output_amplitude.shape[1])), combined_output_amplitude])
            
            combined_label_amplitude = np.vstack([np.zeros((1, combined_label_amplitude.shape[1])), combined_label_amplitude])
            combined_mic_phase = np.vstack([np.zeros((1, combined_mic_phase.shape[1])), combined_mic_phase])


            complex_signal = combined_output_amplitude * np.exp(1j * combined_noisy_phase)
            audio_signal_output = librosa.istft(complex_signal, n_fft=512, hop_length=128, win_length=512, window='hann')
            sf.write(f'{output_dir}/{sound_name}_combined_output.wav', audio_signal_output * 0.5 / max(abs(audio_signal_output)), sr)


            subject = sound_name.split('_')[0]
            base_path = os.path.join(dataset_dir, subject)
            log_path = os.path.join(base_path, "noise_snr_usage_log.txt")
            snr_value = None

            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='cp949') as f:
                    for line in f:
                        match = re.match(r"파일: (.+?) \| .*?SNR: ([-\d]+) dB", line.strip())
                        if match:
                            log_sound_name = match.group(1).replace("_mic.wav", "")
                            if log_sound_name == sound_name:
                                snr_value = int(match.group(2))
                                break

            acc_path = os.path.join(base_path, f"{sound_name}_acc.wav")
            mic_path = os.path.join(base_path, f"{sound_name}_mic.wav")
            noisy_mic_path = os.path.join(base_path, f"{sound_name}_noisy_mic.wav")

            acc_original, sr_acc_original = librosa.load(acc_path, sr=8000)
            mic_original, sr_mic_original = librosa.load(mic_path, sr=16000)
            noisy_mic_original, sr_noisy_original = librosa.load(noisy_mic_path, sr=16000)


            # 16000 Hz
            audio_signal_output_up_sampled = librosa.resample(audio_signal_output, orig_sr=8000, target_sr=16000)
            min_len = min(len(mic_original), len(audio_signal_output_up_sampled))
            mic_original_trimmed = mic_original[:min_len]
            noisy_mic_original_trimmed = noisy_mic_original[:min_len]
            audio_signal_output_up_sampled_trimmed = audio_signal_output_up_sampled[:min_len]

            # 8000 Hz
            mic_original_resampled = librosa.resample(mic_original, orig_sr=16000, target_sr=8000)
            noisy_mic_original_resampled = librosa.resample(noisy_mic_original, orig_sr=16000, target_sr=8000)
            min_len = min(len(mic_original_resampled), len(audio_signal_output))
            mic_original_resampled_trimmed = mic_original_resampled[:min_len]
            noisy_mic_original_resampled_trimmed = noisy_mic_original_resampled[:min_len]
            audio_signal_output_trimmed = audio_signal_output[:min_len]

            output_pesq_wb_score = pesq(16000, mic_original_trimmed, audio_signal_output_up_sampled_trimmed, 'wb')
            output_pesq_nb_score = pesq(sr, mic_original_resampled_trimmed, audio_signal_output_trimmed, 'nb')
            output_stoi_score = stoi(mic_original_resampled_trimmed, audio_signal_output_trimmed, sr, extended=False)
            output_scig_score = compute_csig(mic_original_resampled_trimmed, audio_signal_output_trimmed, sr, norm=False)
            output_cbak_score = compute_cbak(mic_original_resampled_trimmed, audio_signal_output_trimmed, sr, norm=False)
            output_covl_score = compute_covl(mic_original_resampled_trimmed, audio_signal_output_trimmed, sr, norm=False)

            results.append({
                'sound_name': sound_name,
                'input_SNR': snr_value,
                'output_PESQ_WB': output_pesq_wb_score,
                'output_PESQ_NB': output_pesq_nb_score,
                'output_STOI': output_stoi_score,
                'output_CSIG': output_scig_score,
                'output_CBAK': output_cbak_score,
                'output_COVL': output_covl_score
            })          

    logging.info("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_mse)))

    df = pd.DataFrame(results)
    df.to_excel('results/sound_metrics.xlsx', index=False)

    grouped = df.groupby('input_SNR').agg(
        Samples=('sound_name', 'count'),
        Avg_PESQ_WB=('output_PESQ_WB', 'mean'),
        Avg_PESQ_NB=('output_PESQ_NB', 'mean'),
        Avg_STOI=('output_STOI', 'mean'),
        Avg_CSIG=('output_CSIG', 'mean'),
        Avg_CBAK=('output_CBAK', 'mean'),
        Avg_COVL=('output_COVL', 'mean')
    ).reset_index()

    total_avg = df[['output_PESQ_WB', 'output_PESQ_NB', 'output_STOI', 'output_CSIG', 'output_CBAK', 'output_COVL']].mean().to_frame().T
    total_avg.insert(0, 'input_SNR', 'All')
    total_avg.insert(0, 'sound_name', 'Total_Mean')

    save_path = 'results/sound_metrics.xlsx'
    with pd.ExcelWriter(save_path) as writer:
        df.to_excel(writer, sheet_name='All Results', index=False)
        grouped.to_excel(writer, sheet_name='SNR Summary', index=False)
        total_avg.to_excel(writer, sheet_name='Total Summary', index=False)