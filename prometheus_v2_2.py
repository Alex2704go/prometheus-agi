# prometheus_v2_2.py - Версия для Фундаментального Обучения (~15M)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict

# --- Компоненты модели (оптимизированные) ---

class GeometricRotation(nn.Module):
    def __init__(self, dim: int, n_rotations: int = 4):
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
    def __init__(self, dim: int, num_experts: int = 6, top_k: int = 2, expert_mult: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = dim * expert_mult
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x: Tensor) -> Tensor:
        # Упрощенная, но рабочая реализация
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

class PrometheusBlockFT(nn.Module):
    def __init__(self, dim: int, num_experts: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.abel_branch = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.nonabel_branch = nn.Sequential(MixtureOfExperts(dim, num_experts), GeometricRotation(dim))
        self.gate_controller = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())
        
        # Механизм Рефлексии
        self.register_buffer('running_gate_mean', torch.tensor(0.5), persistent=False)
        self.momentum = 0.01
        self.last_gate_value = torch.tensor(0.5)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x_norm = self.norm(x)
        abel_out = self.abel_branch(x_norm)
        nonabel_out = self.nonabel_branch(x_norm)
        
        combined = torch.cat([abel_out, nonabel_out], dim=-1)
        gate_value = self.gate_controller(combined)
        
        current_mean = gate_value.detach().mean()
        if self.training:
            self.running_gate_mean = (1 - self.momentum) * self.running_gate_mean + self.momentum * current_mean
        self.last_gate_value = current_mean
        
        mixed = abel_out * (1 - gate_value) + nonabel_out * gate_value
        return residual + mixed

    def get_balance_loss(self) -> Tensor:
        return (self.last_gate_value - 0.5)**2

    def get_consistency_loss(self) -> Tensor:
        return (self.running_gate_mean - 0.5)**2

class PrometheusFT(nn.Module):
    def __init__(self, vocab_size: int, dim: int, depth: int, num_experts: int, tie_weights: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            PrometheusBlockFT(dim, num_experts) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Связывание весов
        if tie_weights:
            self.lm_head.weight = self.embedding.weight
        
        # Применяем хорошую инициализацию
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None, **kwargs) -> Tuple[Optional[Tensor], Tensor]:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, logits)
    
    # --- Методы для продвинутого обучения (остаются такими же) ---
    
    def get_parameter_groups(self) -> List[Dict]:
        # ... (код без изменений)
        pass

    def get_gate_balance_loss(self) -> Tensor:
        losses = [block.get_balance_loss() for block in self.blocks]
        return torch.stack(losses).mean()

    def get_gate_consistency_loss(self) -> Tensor:
        losses = [block.get_consistency_loss() for block in self.blocks]
        return torch.stack(losses).mean()

    def get_last_avg_gate_value(self) -> Tensor:
        # ... (код без изменений)
        pass

    def topological_regularization(self) -> Tensor:
        # ... (код без изменений)
        pass