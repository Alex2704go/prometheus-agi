# prometheus.py (ФИНАЛЬНАЯ, ПУЛЕНЕПРОБИВАЕМАЯ ВЕРСИЯ)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class GeometricRotation(nn.Module):
    def __init__(self, dim: int, n_rotations: int = 3):
        super().__init__()
        self.generators = nn.ParameterList([
            nn.Parameter(torch.zeros(dim, dim)) 
            for _ in range(n_rotations)
        ])
        self.reset_parameters()
        
    def reset_parameters(self):
        for generator in self.generators:
            nn.init.xavier_uniform_(generator)
            generator.data = 0.5 * (generator.data - generator.data.t())
    
    def forward(self, x: Tensor) -> Tensor:
        for gen in self.generators:
            rot_matrix = torch.matrix_exp(gen)
            x = x @ rot_matrix
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        gate_logits = self.gate(x)
        top_k_weights, top_k_indices = torch.topk(
            gate_logits, k=self.top_k, dim=-1
        )
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        output = torch.zeros_like(x)
        
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i)
            if not expert_mask.any():
                continue
            current_weights = torch.where(
                expert_mask, 
                top_k_weights, 
                torch.zeros_like(top_k_weights)
            ).sum(dim=-1, keepdim=True)
            expert_output = self.experts[i](x)
            output += expert_output * current_weights
        
        return output

class PrometheusBlock(nn.Module):
    def __init__(self, dim: int, num_experts: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.abel_branch = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.nonabel_branch = nn.Sequential(
            MixtureOfExperts(dim, num_experts),
            GeometricRotation(dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        abel_out = self.abel_branch(x)
        nonabel_out = self.nonabel_branch(x)
        combined = torch.cat([abel_out, nonabel_out], dim=-1)
        gate_value = self.gate(combined)
        mixed = abel_out * (1 - gate_value) + nonabel_out * gate_value
        return residual + mixed

class Prometheus(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 dim: int = 512, 
                 depth: int = 12,
                 num_experts: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(*[
            PrometheusBlock(dim, num_experts)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    # Делаем forward максимально совместимым с Hugging Face Trainer
    def forward(
        self, 
        input_ids: Tensor, 
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs  # Это "ловушка" для всех остальных ненужных аргументов
    ) -> Tuple[Optional[Tensor], Tensor]:

        x = self.embedding(input_ids)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        # Если переданы `labels`, Trainer хочет, чтобы мы сами посчитали loss
        if labels is not None:
            # Сдвигаем логиты и метки для предсказания следующего токена
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Вычисляем loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Trainer ожидает, что первым элементом будет loss (если он есть)
        return (loss, logits)
    
    def topological_regularization(self) -> Tensor:
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for name, param in self.named_parameters():
            if 'nonabel_branch' in name and param.dim() > 1:
                norms = torch.norm(param, p=2, dim=1)
                var_norms = torch.var(norms)
                reg_loss += 1 / (var_norms + 1e-6)
                count += 1
        return reg_loss / count if count > 0 else reg_loss