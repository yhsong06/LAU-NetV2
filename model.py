import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, depthwise_separable: bool = False):
        super().__init__()
        if depthwise_separable:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=out_channels,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(stride // 2) + 1,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        B, C, Freq, T = x.size()
        x = x.permute(0, 3, 1, 2).reshape(B * T, C, Freq)
        x = self.net(x)
        C_out, F_out = x.size(1), x.size(2)
        x = x.view(B, T, C_out, F_out).permute(0, 2, 3, 1)
        return x


class USConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=stride, mode="nearest"),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        B, C1, F1, T = x1.shape
        _, C2, F2, _ = x2.shape
        x1 = x1.permute(0, 3, 1, 2).contiguous().reshape(B * T, C1, F1)
        x2 = x2.permute(0, 3, 1, 2).contiguous().reshape(B * T, C2, F2)

        diffF = x2.shape[2] - x1.shape[2]
        if diffF != 0:
            x1 = F.pad(x1, [diffF // 2, diffF - diffF // 2])

        x = torch.cat((x1, x2), dim=1)
        x = self.net(x)
        C_out, F_up = x.shape[1], x.shape[2]
        x = x.view(B, T, C_out, F_up).permute(0, 2, 3, 1).contiguous()
        return x


class USConvLastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=stride, mode="nearest"),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )

    def forward(self, x1, x2):
        B, C1, F1, T = x1.shape
        _, C2, F2, _ = x2.shape
        x1 = x1.permute(0, 3, 1, 2).contiguous().reshape(B * T, C1, F1)
        x2 = x2.permute(0, 3, 1, 2).contiguous().reshape(B * T, C2, F2)

        diffF = x2.shape[2] - x1.shape[2]
        if diffF != 0:
            x1 = F.pad(x1, [diffF // 2, diffF - diffF // 2])

        x = torch.cat((x1, x2), dim=1)
        x = self.net(x)
        C_out, F_up = x.shape[1], x.shape[2]
        x = x.view(B, T, C_out, F_up).permute(0, 2, 3, 1).contiguous()
        return x


class HarmonicFiLMFGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional=True):
        super().__init__()
        self.harmonic_reduction = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=5, stride=4, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.gamma_beta_generator = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, in_channels * 2, kernel_size=1),
        )
        self.gru = nn.GRU(in_channels, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_size * (2 if bidirectional else 1), out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, harmonic_mask):
        B, C, Freq, T = x.shape
        hm = harmonic_mask  # [B,1,Freq,T]
        hm = hm.permute(0, 3, 1, 2).contiguous().reshape(B * T, 1, Freq)
        feat = self.harmonic_reduction(hm)
        gamma_beta = self.gamma_beta_generator(feat)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(B, T, C, Freq).permute(0, 2, 3, 1).contiguous()
        beta = beta.view(B, T, C, Freq).permute(0, 2, 3, 1).contiguous()

        x_mod = x * (1 + gamma) + beta
        x_gru = x_mod.permute(0, 3, 2, 1).contiguous().reshape(B * T, Freq, C)
        out, _ = self.gru(x_gru)
        out = out.transpose(1, 2)
        out = self.proj(out)
        out = out.view(B, T, -1, Freq).permute(0, 2, 3, 1).contiguous()
        return out


class VADFiLMTGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional=True):
        super().__init__()
        self.gamma_beta_generator = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, in_channels * 2, kernel_size=1),
        )
        self.gru = nn.GRU(in_channels, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_size * (2 if bidirectional else 1), out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, vad_mask):
        B, C, Freq, T = x.shape
        vm = vad_mask  # [B,1,Freq,T]
        vad_global = vm.mean(dim=2)  # [B,1,T]
        gamma_beta = self.gamma_beta_generator(vad_global)  # [B,2C,T]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # [B,C,T]
        gamma = gamma.unsqueeze(2)  # [B,C,1,T]
        beta = beta.unsqueeze(2)    # [B,C,1,T]

        x_mod = x * (1 + gamma) + beta
        x_gru = x_mod.permute(0, 2, 3, 1).contiguous().reshape(B * Freq, T, C)
        out, _ = self.gru(x_gru)
        out = out.transpose(1, 2)
        out = self.proj(out)
        out = out.view(B, Freq, -1, T).permute(0, 2, 1, 3).contiguous()
        return out


class LAUNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DSConvBlock(2, 16, 5, 2, depthwise_separable=False)
        self.enc2 = DSConvBlock(16, 32, 5, 2, depthwise_separable=True)
        self.enc3 = DSConvBlock(32, 64, 5, 2, depthwise_separable=True)

        self.harmonic_film_fgru = HarmonicFiLMFGRUBlock(64, 32, 32, bidirectional=False)
        self.vad_film_tgru = VADFiLMTGRUBlock(32, 32, 32, bidirectional=False)

        self.dec1 = USConvBlock(96, 32, 5, 2)
        self.dec2 = USConvBlock(64, 16, 5, 2)
        self.dec3 = USConvLastBlock(32, 1, 5, 2)

    def forward(self, x):
        if x.size(1) != 4:
            raise ValueError(f"LAUNetV2 expects 4-channel input [noisyAM, ACC, harmonic, VAD], got C={x.size(1)}")

        x_in = x[:, 0:2, :, :]     # [B,2,F,T]
        harmonic = x[:, 2:3, :, :] # [B,1,F,T]
        vad = x[:, 3:4, :, :]      # [B,1,F,T]

        x1 = self.enc1(x_in)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x4 = self.harmonic_film_fgru(x3, harmonic)
        x5 = self.vad_film_tgru(x4, vad)

        x6 = self.dec1(x5, x3)
        x7 = self.dec2(x6, x2)
        x8 = self.dec3(x7, x1)
        return x8


if __name__ == "__main__":
    net = LAUNetV2()
    total_params = sum(p.numel() for p in net.parameters())
    x = torch.randn(1, 4, 256, 4)
    y = net(x)
    print("total params:", total_params)
    print("input_shape:", x.shape)
    print("output_shape:", y.shape)