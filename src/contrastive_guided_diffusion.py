"""
Reference:
- https://github.com/YiDongOuYang/Contrastive-Guided-Diffusion-Process.git
"""

from torch import nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from enum import Enum


class MLPBlock(nn.Module):
    def __init__(self, input_size=640, hidden_size=128):
        super().__init__()
        self.d1 = nn.Linear(input_size, hidden_size, bias=False)
        self.d2 = nn.Linear(input_size, hidden_size, bias=False)
        self.d3 = nn.Linear(hidden_size, input_size, bias=False)
    
    def reset_parameters(self):
        self.d1.reset_parameters()
        self.d2.reset_parameters()
        self.d3.reset_parameters()

    def forward(self, input):
        out = self.d3(torchF.silu(self.d1(input)) * self.d2(input))
        return out

class MLP(nn.Module):
    def __init__(self, input_size=640, hidden_size=128):
        super(MLP, self).__init__()
        self.layer0 = MLPBlock(input_size, hidden_size)
        self.layer1 = MLPBlock(input_size, hidden_size)
        self.layer2 = MLPBlock(input_size, hidden_size)

    def reset_parameters(self):
        self.layer0.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
    
    def forward(self, input: torch.Tensor):
        out = input
        out = self.layer0(out) + out
        out = self.layer1(out) + out
        out = self.layer2(out) + out
        return out

class Estimator(Enum):
    hard = 0
    soft = 1


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        tau_plus: float = 0.1,
        beta: float = 1.0,
        temperature: float = 10,
        estimator: Estimator = Estimator.hard,
    ):
        super().__init__()

        self.tau_plus = tau_plus
        self.beta = beta
        self.temperature = temperature

        if isinstance(estimator, str):
            estimator = estimator.lower()
            if estimator == "hard":
                estimator = Estimator.hard
            elif estimator == "soft":
                estimator = Estimator.soft
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

        self.estimator = estimator

    @torch.no_grad()
    def negative_mask(self, batch_size):
        mask = torch.ones((batch_size, 2 * batch_size), dtype=torch.bool)
        if batch_size != 1:
            for i in range(batch_size):
                mask[i, i] = 0
                mask[i, i + batch_size] = 0
        else:
            mask[0, 1] = 0
        mask = torch.cat([mask, mask], dim=0)
        return mask

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        # Compute cosine similarity
        batch_size = x_1.size(0) 
        # print(f"contrastive loss's x_1 batch size: {batch_size} and x_1 shape {x_1.shape}")
        # print(f"contrastive loss's x_2 batch size: {x_2.size(0)} and x_2 shape {x_2.shape}")
        # negative score
        out = torch.cat([x_1, x_2], dim=0)
        neg = torch.exp(torchF.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1) / self.temperature)

        mask = self.negative_mask(batch_size).to(x_1.device)
        # print(f"size of mask: {mask.shape}")
        neg = neg.masked_select(mask).view(2 * batch_size, -1)
        # print(f"contrastive loss's masked negative shape: {neg.shape}")
        # positive score
        pos = torch.exp(torchF.cosine_similarity(x_1.unsqueeze(1), x_2.unsqueeze(0), dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if self.estimator == Estimator.hard:
            N = max(batch_size * 2 - 2, 2)
            neg = torch.clamp(neg, min=1e-6)
            imp = (self.beta * neg.log()).exp()  # ??
            reweight_neg = (imp * neg).sum(dim=-1, keepdim=True) / \
                imp.mean(dim=-1, keepdim=True)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)

            # constrain (optinal)
            Ng = torch.clamp(Ng, min=N*np.e**(-1 / self.temperature))
        elif self.estimator == Estimator.soft:
            Ng = neg.sum(dim=-1)
        else:
            raise NotImplementedError(
                f"Estimator {self.estimator} is not supported currently.")
        
        ratio = torch.clamp(pos / (pos + Ng), min=1e-6)

        loss = - torch.log(ratio).mean()  # + 0.1* torchF.mse_loss(x_1, x_2)

        # print(cos_sim_12)
        # print(f"\033[32mloss: {loss}, {loss_2.item()}; {torch.norm(x_1)} pos {pos.mean()}, neg {Ng.mean()}\033[0m")

        # check nan
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss




class ContrastiveGuidedDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        classifier: nn.Module,
        n_classes: int,
        batch_size: int,
        img_shape: tuple,
        optim_lr=1e-2,
        enable_drift=True,
        drift_counter_min=2,
        drift_counter_max=10000,
        tau_plus: float = 0.1,
        beta: float = 1.0,
        temperature: float = 10,
        delta_grad_mul: float = 5000, # not used anymore
        estimator: Estimator = Estimator.hard,
    ):
        """
        Requires:
        - model:
        - classifier:
        - n_classes:
        - optim_lr: the learning rate for optimizer
        - enable_drift: 
        - drift_counter_min:
        - drift_counter_max:
            compute the drift as `x + drift` when
            drift_counter_min <= self.drift_counter < self.dirft_counter_max

            self.drift_counter start from 0
        """
        super().__init__()
        self.model = model
        self.optim_lr = optim_lr

        self.classifier = classifier.eval()
        self.embed_net = MLP(n_classes, 2*n_classes)
        self.optimizer = torch.optim.Adam(self.embed_net.parameters(), lr=self.optim_lr)

        self.contrastive_loss = ContrastiveLoss(
            tau_plus=tau_plus,
            beta=beta,
            temperature=temperature,
            estimator=estimator,
        )

        self.xt_cache = None
        self.enable_drift = enable_drift

        self.drift_counter = 0
        self.drift_counter_min = drift_counter_min
        self.drift_counter_max = drift_counter_max

        self.delta_grad_mul = delta_grad_mul

    def reset_drift_counter(self):
        self.drift_counter = 0
        self.embed_net.reset_parameters()
        del self.optimizer
        self.optimizer = torch.optim.Adam(self.embed_net.parameters(), lr=self.optim_lr)

    def acquisition_grad(self, x: torch.Tensor, x0: torch.Tensor):
        x = x.detach().clone()
        x0 = x0.detach().clone()

        with torch.no_grad():
            _x_0 = self.classifier(x0)  

        with torch.enable_grad():
            delta = torch.zeros_like(x, requires_grad=True)
            _x_0 = self.embed_net(_x_0) # type: torch.Tensor
            _x_d = self.classifier(x + delta)  # type: torch.Tensor
            _x_d = self.embed_net(_x_d)

            loss = self.contrastive_loss(_x_0, _x_d)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return delta.grad.detach() * self.delta_grad_mul

    def forward(self, x: torch.Tensor, t, *, enable_drift=None):
        # self.xt_cache.append(x)
        # print("\033[32mcontrastive model forward\033[0m", self.drift_counter)

        out = self.model.forward(x, t)

        # print("\033[32mcontrastive model forward\033[0m", out.shape, x.shape)

        if enable_drift is None:
            enable_drift = self.enable_drift
        if self.xt_cache is not None:
            if self.xt_cache.shape != x.shape:
                self.xt_cache = None
                self.reset_drift_counter()
        
        if self.xt_cache is None:
            enable_drift = False

        if enable_drift and (self.drift_counter_min <= self.drift_counter) and (self.drift_counter < self.drift_counter_max):
            delta = self.acquisition_grad(x, self.xt_cache)
            # print(f"[{self.drift_counter}] delta/out {torch.norm(delta, p=2).item() / torch.norm(out[:, 0:x.size(1)], p=2).item()}")
            # it is possible that the module contains learned sigma,
            # so `out` contains (mean, val)
            # only takes the channels from 0 to x.size(1) into drift
            out[:, 0:x.size(1)] = out[:, 0:x.size(1)] + delta

        self.xt_cache = x.detach().clone()

        self.drift_counter += 1
        return out
