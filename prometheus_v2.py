# prometheus_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict, Iterator
import numpy as np

# --- Компоненты модели (остаются без изменений) ---

class GeometricRotation(nn.Module):
    def __init__(self, dim: int, n_rotations: int = 3):
        super().__init__()
        self.generators = nn.ParameterList([
            nn.Parameter(torch.zeros(dim, dim)) for _ in range(n_rotations)
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
                nn.Linear(dim, dim * 2), # Уменьшим размер для экономии
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        gate_logits = self.gate(x)
        top_k_weights, top_k_indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        output = torch.zeros_like(x)
        
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i)
            if not expert_mask.any():
                continue
            current_weights = torch.where(expert_mask, top_k_weights, torch.zeros_like(top_k_weights)).sum(dim=-1, keepdim=True)
            expert_output = self.experts[i](x)
            output += expert_output * current_weights
        
        return output

# --- Обновленный PrometheusBlock с механизмами активации ---

class PrometheusBlockV2(nn.Module):
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
        # Улучшенный гейт, который мы можем анализировать
        self.gate_controller = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        
        # "Крючки" для анализа
        self.gate_values_history = []
        self.gate_balance_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x_norm = self.norm(x)
        
        abel_out = self.abel_branch(x_norm)
        nonabel_out = self.nonabel_branch(x_norm)
        
        combined = torch.cat([abel_out, nonabel_out], dim=-1)
        gate_value = self.gate_controller(combined)
        
        # Сохраняем значения гейта для анализа (только во время eval)
        if not self.training:
            self.gate_values_history.append(gate_value.detach().cpu())

        # Считаем балансировочный лосс
        self.gate_balance_loss = torch.mean((gate_value - 0.5)**2)
        
        mixed = abel_out * (1 - gate_value) + nonabel_out * gate_value
        return residual + mixed

    def reset_gate_analysis(self):
        self.gate_values_history = []

# --- Обновленная модель PrometheusV2 ---

class PrometheusV2(nn.Module):
    def __init__(self, vocab_size: int, dim: int, depth: int, num_experts: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            PrometheusBlockV2(dim, num_experts) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self, 
        input_ids: Tensor, 
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Optional[Tensor], Tensor]:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, logits)
    
    # --- Методы для продвинутого обучения ---
    
    def get_parameter_groups(self) -> List[Dict]:
        """Разделяет параметры на группы для дифференцированного обучения."""
        abel_params, nonabel_params, gate_params, other_params = [], [], [], []
        
        for name, param in self.named_parameters():
            if 'abel_branch' in name:
                abel_params.append(param)
            elif 'nonabel_branch' in name:
                nonabel_params.append(param)
            elif 'gate_controller' in name:
                gate_params.append(param)
            else:
                other_params.append(param)
        
        return [
            {'name': 'abel', 'params': abel_params},
            {'name': 'nonabel', 'params': nonabel_params},
            {'name': 'gate', 'params': gate_params},
            {'name': 'other', 'params': other_params},
        ]

    def get_gate_balance_loss(self) -> Tensor:
        """Собирает балансировочный лосс со всех блоков."""
        total_gate_loss = torch.tensor(0.0, device=self.embedding.weight.device)
        for block in self.blocks:
            total_gate_loss += block.gate_balance_loss
        return total_gate_loss / len(self.blocks)

    def topological_regularization(self) -> Tensor:
        """Регуляризация для неабелевой ветки."""
        reg_loss = torch.tensor(0.0, device=self.embedding.weight.device)
        count = 0
        for name, param in self.named_parameters():
            if 'nonabel_branch' in name and param.dim() > 1:
                norms = torch.norm(param, p=2, dim=1)
                var_norms = torch.var(norms)
                if not torch.isnan(var_norms):
                    reg_loss += 1 / (var_norms + 1e-6)
                    count += 1
        return reg_loss / count if count > 0 else reg_loss