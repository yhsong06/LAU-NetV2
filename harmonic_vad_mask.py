import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from preprocessing import load_segmented_data, load_segmented_data_val, load_segmented_data_test


class HarmonicEstimationBlock(nn.Module):
    def __init__(self, peak_distance=2, max_peaks=5, freq_margin=3):
        super().__init__()
        self.peak_distance = peak_distance
        self.max_peaks = max_peaks
        self.freq_margin = freq_margin

    def extract_f0(self, magnitudes_2d: torch.Tensor, max_power: float):
        z = magnitudes_2d.clone()
        min_peak_indices = []

        for col in range(z.shape[1]):
            column = z[:, col]
            peak_indices = []

            for _ in range(self.max_peaks):
                max_value, max_index = torch.max(column[1:], dim=0)
                max_index = max_index + 1

                if max_value.item() > max_power:
                    peak_indices.append(int(max_index.item()))
                    column[max_index] = -float("inf")
                else:
                    break

            min_peak_indices.append(min(peak_indices) if peak_indices else 0)

        return min_peak_indices

    def harmonic_estimation(self, magnitudes_2d: torch.Tensor, max_power: float):
        binmask = torch.full_like(magnitudes_2d, 0.5, dtype=torch.float32)
        f0_indices = self.extract_f0(magnitudes_2d, max_power=max_power)

        for col, f0_index in enumerate(f0_indices):
            if f0_index <= 0:
                continue

            pitch_index = f0_index
            for i in range(pitch_index, magnitudes_2d.shape[0] - (self.freq_margin + 1), pitch_index):
                lo = i - self.freq_margin
                hi = i + self.freq_margin
                for k in range(lo, hi + 1):
                    if 0 <= k < magnitudes_2d.shape[0]:
                        distance = abs(k - i)
                        val = max(1 - 0.5 * (distance / self.freq_margin), 0.5)
                        binmask[k, col] = max(binmask[k, col], val)

        return binmask


def _ensure_NFT(acc):
    acc = np.asarray(acc)
    if acc.ndim == 4:
        if acc.shape[1] == 1:
            acc = acc[:, 0, :, :]
        else:
            acc = acc[:, 0, :, :]
    if acc.ndim != 3:
        raise ValueError(f"Expected acc as [N,F,T] or [N,1,F,T], got shape={acc.shape}")
    return acc


def compute_vad_mask_from_acc_mag(
    acc_mag_NFT: np.ndarray,
    non_speech_quantile: float = 0.2,
    dc_exclude: int = 1,
):
    N, F, T = acc_mag_NFT.shape
    vad_masks = np.zeros((N, 1, F, T), dtype=np.float32)

    for i in range(N):
        mag = acc_mag_NFT[i]  # [F,T]
        mag_used = mag[dc_exclude:, :] if F > dc_exclude else mag

        rms_t = np.sqrt(np.mean(np.square(mag_used), axis=0) + 1e-12)  # [T]

        k = max(1, int(np.floor(T * float(non_speech_quantile))))
        idx_sorted = np.argsort(rms_t)
        non_speech_idx = idx_sorted[:k]
        thr = float(np.mean(rms_t[non_speech_idx]) + 1e-12)

        vad_t = (rms_t > thr).astype(np.float32)  # [T]
        vad_masks[i, 0, :, :] = vad_t[None, :].repeat(F, axis=0)

    return vad_masks


def compute_harmonic_masks(
    acc_mag_NFT: np.ndarray,
    max_power: float = 0.5,
):
    N, F, T = acc_mag_NFT.shape
    harmonic_block = HarmonicEstimationBlock()
    harmonic_masks = np.zeros((N, 1, F, T), dtype=np.float32)

    for i in range(N):
        x = torch.tensor(acc_mag_NFT[i], dtype=torch.float32)  # [F,T]
        hm = harmonic_block.harmonic_estimation(x, max_power=max_power)  # [F,T]
        harmonic_masks[i, 0] = hm.cpu().numpy().astype(np.float32)

        if (i + 1) % 1000 == 0:
            print(f"[Harmonic] {i + 1}/{N}")

    return harmonic_masks


def _load_acc_by_split(save_dir: str, split: str):
    if split == "train":
        acc, _, _ = load_segmented_data(save_dir=save_dir)
        return acc
    if split == "val":
        acc, _, _ = load_segmented_data_val(save_dir=save_dir)
        return acc
    if split == "test":
        _, acc, _, _, _, _, _ = load_segmented_data_test(save_dir=save_dir)
        return acc
    raise ValueError(f"Unknown split: {split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", type=str, default="./segmented_data")
    ap.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="train")
    ap.add_argument("--max_power", type=float, default=0.5)
    ap.add_argument("--non_speech_quantile", type=float, default=0.2)
    args = ap.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for sp in splits:
        print(f"\n=== Split: {sp} ===")
        acc = _load_acc_by_split(save_dir, sp)
        acc = _ensure_NFT(acc)

        harmonic_masks = compute_harmonic_masks(acc, max_power=args.max_power)
        vad_masks = compute_vad_mask_from_acc_mag(acc, non_speech_quantile=args.non_speech_quantile)

        out_h = os.path.join(save_dir, f"harmonic_masks_{sp}.npy")
        out_v = os.path.join(save_dir, f"vad_masks_{sp}.npy")

        np.save(out_h, harmonic_masks)
        np.save(out_v, vad_masks)

        print(f"[DONE] harmonic: {out_h}  shape={harmonic_masks.shape}")
        print(f"[DONE] vad     : {out_v}  shape={vad_masks.shape}")


if __name__ == "__main__":
    main()