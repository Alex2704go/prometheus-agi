import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class DynamicGeometricRotation(nn.Module):
    """Динамические вращения с входно-зависимыми параметрами"""
    def __init__(self, dim: int, hidden_dim: int = 128, n_rotations: int = 3):
        super().__init__()
        self.dim = dim
        self.n_rotations = n_rotations
        
        # Нейросеть для генерации параметров вращения
        self.rotation_generator = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_rotations * dim * dim)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Генерируем параметры вращения на основе контекста
        params = self.rotation_generator(x.mean(dim=1))  # [batch_size, n_rotations * dim * dim]
        params = params.view(batch_size, self.n_rotations, self.dim, self.dim)
        
        # Применяем последовательность вращений
        for i in range(self.n_rotations):
            # Создаем антисимметричную матрицу
            generator = params[:, i]  # [batch_size, dim, dim]
            generator = 0.5 * (generator - generator.transpose(1, 2))
            
            # Вычисляем матрицу вращения
            rot_matrix = torch.matrix_exp(generator)
            
            # Применяем вращение
            x = torch.einsum('bnd,bde->bne', x, rot_matrix)
            
        return x

class EfficientMoE(nn.Module):
    """Оптимизированная реализация смеси экспертов"""
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2, capacity_factor: float = 1.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.dim = dim
        
        # Эксперты
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        
        # Маршрутизатор
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.dim)  # [batch_size * seq_len, dim]
        
        # Вычисляем логиты маршрутизации
        gate_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Выбираем top_k экспертов
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(gate_logits, dim=-1), 
            k=self.top_k, 
            dim=-1
        )
        
        # Создаем маску для экспертов
        expert_mask = torch.zeros_like(gate_logits, dtype=torch.bool)
        expert_mask.scatter_(1, top_k_indices, True)
        
        # Рассчитываем емкость каждого эксперта
        capacity = int(self.capacity_factor * len(x_flat) / self.num_experts)
        
        # Собираем токены для каждого эксперта
        expert_inputs = []
        expert_weights = []
        for i in range(self.num_experts):
            # Токены, назначенные текущему эксперту
            token_indices = torch.where(expert_mask[:, i])[0]
            if len(token_indices) > capacity:
                # Выбираем токены с наибольшими весами
                _, indices = top_k_weights[token_indices, torch.where(top_k_indices[token_indices] == i)[1]].topk(capacity)
                token_indices = token_indices[indices]
            
            expert_inputs.append(x_flat[token_indices] if len(token_indices) > 0 else None)
            expert_weights.append(top_k_weights[token_indices, torch.where(top_k_indices[token_indices] == i)[1]] 
                                 if len(token_indices) > 0 else None)
        
        # Обрабатываем токены экспертами
        expert_outputs = []
        for i, (expert, inputs) in enumerate(zip(self.experts, expert_inputs)):
            if inputs is not None:
                expert_outputs.append((expert(inputs), expert_weights[i], inputs))
            else:
                expert_outputs.append(None)
        
        # Собираем результаты
        output_flat = torch.zeros_like(x_flat)
        for i, out_data in enumerate(expert_outputs):
            if out_data is None:
                continue
                
            expert_out, weights, indices = out_data
            weighted_out = expert_out * weights.unsqueeze(-1)
            output_flat[indices] += weighted_out
        
        return output_flat.view(batch_size, seq_len, self.dim)

class PrometheusBlock(nn.Module):
    """Усовершенствованный блок когнитивного метаболизма"""
    def __init__(self, dim: int, num_experts: int = 8, 
                 gate_analysis: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.abel_branch = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.nonabel_branch = nn.Sequential(
            EfficientMoE(dim, num_experts),
            DynamicGeometricRotation(dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        self.gate_analysis = gate_analysis
        self.gate_values = [] if gate_analysis else None
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        
        # Параллельные пути обработки
        abel_out = self.abel_branch(x)
        nonabel_out = self.nonabel_branch(x)
        
        # Комбинируем выходы для гейта
        combined = torch.cat([abel_out, nonabel_out], dim=-1)
        gate_value = self.gate(combined)
        
        # Сохраняем значения гейта для анализа
        if self.gate_analysis:
            self.gate_values.append(gate_value.detach().cpu())
        
        # Смешиваем пути
        mixed = abel_out * (1 - gate_value) + nonabel_out * gate_value
        
        return residual + mixed
    
    def reset_gate_analysis(self):
        """Сброс накопленных значений гейта"""
        if self.gate_analysis:
            self.gate_values = []

class Prometheus(nn.Module):
    """Производственная версия модели Prometheus"""
    def __init__(self, 
                 vocab_size: int, 
                 dim: int = 512, 
                 depth: int = 12,
                 num_experts: int = 8,
                 gate_analysis: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            PrometheusBlock(dim, num_experts, gate_analysis)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        
    def forward(self, input_ids: Tensor) -> Tensor:
        # Встраивание токенов
        x = self.embedding(input_ids)
        
        # Применение блоков
        for block in self.blocks:
            x = block(x)
        
        # Финальные преобразования
        x = self.norm(x)
        return self.lm_head(x)
    
    def topological_regularization(self) -> Tensor:
        """Улучшенная топологическая регуляризация"""
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        
        for name, param in self.named_parameters():
            if 'nonabel_branch' in name and param.dim() > 1:
                # Вычисляем L2 норму каждого нейрона
                if param.dim() == 2:  # Линейные слои
                    norms = torch.norm(param, p=2, dim=1)
                elif param.dim() == 4:  # Сверточные веса
                    norms = torch.norm(param, p=2, dim=(1, 2, 3))
                else:  # Другие параметры
                    continue
                    
                # Мера разнообразия: логарифм дисперсии норм
                var_norms = torch.var(norms)
                reg_loss += -torch.log(var_norms + 1e-6)  # Максимизируем разнообразие
                count += 1
        
        return reg_loss / count if count > 0 else reg_loss
    
    def get_gate_analysis(self) -> dict:
        """Анализ поведения гейта по всем блокам"""
        if not self.blocks[0].gate_analysis:
            raise ValueError("Gate analysis not enabled")
            
        results = {}
        for i, block in enumerate(self.blocks):
            if block.gate_values:
                values = torch.cat(block.gate_values, dim=0)
                results[f'block_{i}'] = {
                    'mean': values.mean().item(),
                    'std': values.std().item(),
                    'min': values.min().item(),
                    'max': values.max().item(),
                    'hist': torch.histc(values, bins=10, min=0, max=1).tolist()
                }
                block.reset_gate_analysis()
                
        return results

# Пример использования с распределенным обучением
def train_distributed(model, dataloader, optimizer, device, epochs=10):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Основная функция потерь
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=0  # Игнорируем padding
            )
            
            # Топологическая регуляризация
            topo_reg = model.module.topological_regularization()
            
            # Динамический вес регуляризации
            progress = batch_idx / len(dataloader)
            topo_weight = 0.1 * (1 + torch.cos(2 * torch.pi * progress))
            
            total_loss = loss + topo_weight * topo_reg
            
            # Обратное распространение
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Логирование каждые 100 батчей
            if batch_idx % 100 == 0:
                gate_analysis = model.module.get_gate_analysis() if model.module.blocks[0].gate_analysis else {}
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | Topo: {topo_reg.item():.4f} | "
                      f"Gate Mean: {gate_analysis.get('block_0', {}).get('mean', 0):.4f}")
    

    return model
