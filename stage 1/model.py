"""
LSTM-наукастер: prior → h₀, encoder LSTM, one-shot decoder.

Архитектура (из PLAN.md §4):
  - prior_proj:  Linear(2 → hidden)        — задаёт начальное скрытое состояние
  - encoder:     LSTM(input=2, hidden, num_layers, batch_first)
  - decoder:     Linear(hidden → n_max)    — one-shot, маска на паддинг

  Loss = MSE(preds·mask, target·mask) + λ · (max(preds) − log10(J_max_real))²
"""

from __future__ import annotations

import torch
from torch import nn

from dataset import N_MAX


class LSTMNowcaster(nn.Module):
    def __init__(self,
                 input_size:  int = 2,
                 prior_size:  int = 2,
                 hidden_size: int = 64,
                 num_layers:  int = 2,
                 n_max:       int = N_MAX,
                 dropout:     float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.n_max       = n_max

        self.prior_proj = nn.Linear(prior_size, num_layers * hidden_size)
        self.encoder    = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(hidden_size, n_max)

    def forward(self, encoder_input: torch.Tensor, encoder_mask: torch.Tensor,
                prior: torch.Tensor) -> torch.Tensor:
        """
        encoder_input : [B, K, 2]
        encoder_mask  : [B, K]   1 на наблюдённых шагах, 0 на padding
        prior         : [B, 2]
        Возвращает: preds [B, N_max]
        """
        B = encoder_input.size(0)

        # prior → h₀  (num_layers × B × hidden)
        h0 = self.prior_proj(prior).view(B, self.num_layers, self.hidden_size)
        h0 = h0.transpose(0, 1).contiguous()
        c0 = torch.zeros_like(h0)

        # Pack по реальным длинам, чтобы LSTM не «видел» паддинг
        lengths = encoder_mask.sum(dim=1).long().cpu().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            encoder_input, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_last, _) = self.encoder(packed, (h0, c0))
        h_enc = h_last[-1]   # [B, hidden]

        return self.decoder(h_enc)   # [B, N_max]


def nowcaster_loss(preds: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor,
                   log_J_max: torch.Tensor, lam_peak: float = 1.0
                   ) -> tuple[torch.Tensor, dict]:
    """
    preds       : [B, N_max]
    target      : [B, T] где T ≤ N_max — ground-truth log_J после k шагов
    target_mask : [B, T] — 1 где есть данные
    log_J_max   : [B]    — реальное log10(J_max) события
    """
    B, T = target.shape
    preds_t = preds[:, :T]   # обрезаем до длины target

    sq    = (preds_t - target) ** 2 * target_mask
    denom = target_mask.sum().clamp_min(1.0)
    L_profile = sq.sum() / denom

    # Вершина прогноза по первым T шагам (или по всему N_max — берём по T для честности)
    pred_max = preds_t.max(dim=1).values   # [B]
    L_peak   = ((pred_max - log_J_max) ** 2).mean()

    loss = L_profile + lam_peak * L_peak
    return loss, {"L_profile": float(L_profile.item()), "L_peak": float(L_peak.item())}
