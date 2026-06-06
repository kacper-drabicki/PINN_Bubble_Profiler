import torch
import numpy as np
from dataclasses import dataclass
from functools import partial
from math import sqrt

# Physical constants (GeV)
vEW = 246.22 #0T Higgs VEV
Mh = 125.1 #0T Higgs mass
Mw = 80.385 #0T W mass
Mz = 91.1876 #0T Z mass
Mt = 173.34 #0T top quark mass

# Dimensionless couplings derived from above constants
g = (2.0*Mw)/vEW # SU2L gauge coupling
gPrime = sqrt((4.0*(Mz**2 - Mw**2))/(vEW**2)) # U1Y gauge coupling
h_t = (sqrt(2)*Mt)/vEW # Top quark coupling
lambda_h = (Mh**2)/(2.0*vEW**2) # Higgs quartic coupling

class Potential:
    
    def __init__(self, Tc: float, lambda_m: float, lambda_s: float):
        self.Tc = Tc
        self.lambda_m = lambda_m
        self.lambda_s = lambda_s
        
        # Core physical parameter computation moved directly to initialization
        self.c_h = (1.0 / 48.0) * (9.0 * g**2 + 3.0 * gPrime**2 + 2.0 * (6.0 * h_t + 12.0 * lambda_h + lambda_m))
        self.c_s = (1.0 / 12.0) * (2.0 * lambda_m + 3.0 * lambda_s)
        
        self.mu_2_h = lambda_h * (vEW**2)
        v2_tc = vEW**2 - (self.c_h / lambda_h) * (Tc**2)
        self.mu_2_s = -v2_tc * sqrt(lambda_h * lambda_s) - self.c_s * (Tc**2)

    def v(self, T: float):
        return sqrt(vEW**2 - (self.c_h / lambda_h) * (T**2))

    def w(self, T: float):
        return sqrt(-(self.mu_2_s + self.c_s * (T**2)) / self.lambda_s)

    def V(self, X, T: float):
        h, s = X[:, 0:1], X[:, 1:2]
        term_h = -0.5 * self.mu_2_h * h**2 + 0.25 * lambda_h * h**4
        term_s = 0.5 * self.mu_2_s * s**2 + 0.25 * self.lambda_s * s**4
        term_mixed = 0.25 * self.lambda_m * (s * h) ** 2
        term_thermal = 0.5 * (self.c_h * h**2 + self.c_s * s**2) * T**2
        return term_h + term_s + term_mixed + term_thermal

    def dV(self, X, T: float):
        h, s = X[:, 0:1], X[:, 1:2]
        t2 = (T**2) / (vEW**2)
        mu_h = self.mu_2_h / (vEW**2)
        mu_s = self.mu_2_s / (vEW**2)

        dv_dh = -mu_h * h + lambda_h * h**3 + 0.5 * self.lambda_m * s**2 * h + self.c_h * h * t2
        dv_ds = mu_s * s + self.lambda_s * s**3 + 0.5 * self.lambda_m * h**2 * s + self.c_s * s * t2
        
        # check if X is torch.tensor or np.array so that we can return the same type
        if isinstance(X, torch.Tensor):
            return torch.concat((dv_dh, dv_ds), dim=1)
        elif isinstance(X, np.ndarray):
            return np.concatenate((dv_dh, dv_ds), axis=1)
        else:
            raise ValueError("Unsupported type for X. Expected torch.Tensor or np.ndarray.")
    
class Loss:
    
    def __init__(self, potential: Potential, T: float, is_pretrain: bool = True):
        self.potential = potential
        self.T = T
        self.is_pretrain = is_pretrain

    def __call__(self, r: torch.Tensor, X: torch.Tensor):
        h, s = X[:, 0:1], X[:, 1:2]

        h_r = torch.autograd.grad(h, r, torch.ones_like(h), create_graph=True)[0]
        h_rr = torch.autograd.grad(h_r, r, torch.ones_like(h_r), create_graph=True)[0]

        s_r = torch.autograd.grad(s, r, torch.ones_like(s), create_graph=True)[0]
        s_rr = torch.autograd.grad(s_r, r, torch.ones_like(s_r), create_graph=True)[0]

        # Loss fizyczny 
        dV_values = self.potential.dV(X, self.T)
        loss_physics_h = torch.mean((h_rr + 2 / r * h_r - dV_values[:,0:1])**2)
        loss_physics_s = torch.mean((s_rr + 2 / r * s_r - dV_values[:,1:2])**2)
   
        h_limit = h[-1:]
        s_limit = s[-1:]

        if self.is_pretrain:
            h_zero = h[0:1]
            s_zero = s[0:1]
            loss_boundary_zero_h = torch.mean((h_zero - self.potential.v(self.T)/vEW)**2)
            loss_boundary_zero_s = torch.mean((s_zero)**2)
        else:
            h_r_zero = h_r[0:1]
            s_r_zero = s_r[0:1]
            loss_boundary_zero_h = torch.mean(h_r_zero**2)
            loss_boundary_zero_s = torch.mean(s_r_zero**2)
        
        loss_boundary_max_h = torch.mean((h_limit)**2)
        loss_boundary_max_s = torch.mean((s_limit - self.potential.w(self.T)/vEW)**2)

        loss = (loss_physics_h + loss_boundary_zero_h + loss_boundary_max_h +
                loss_physics_s + loss_boundary_zero_s + loss_boundary_max_s)

        return loss


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    modelPath: str = "saved_models/singlet.pth" 
    output_dim: int = 2
    # fizyka
    Tc: float = 110.0
    lambda_m: float = 1.5
    lambda_s: float = 0.65
    T: float = 105.0
    r_max: float = 90.0
    # pretrain
    pretrain_epochs: int = 50000
    pretrain_loss_fn: callable = Loss(Potential(Tc, lambda_m, lambda_s), T=T, is_pretrain=True)
    pretrain_optimizer: callable = partial(torch.optim.Adam, lr=1e-2)
    pretrain_scheduler: callable = partial(torch.optim.lr_scheduler.OneCycleLR, 
                                           max_lr=1e-2, 
                                           total_steps=pretrain_epochs)
    # finetune
    finetune_epochs: int = 200000
    finetune_loss_fn: callable = Loss(Potential(Tc, lambda_m, lambda_s), T=T, is_pretrain=False)
    finetune_optimizer: callable = partial(torch.optim.Adam, lr=1e-2)
    finetune_scheduler: callable = partial(torch.optim.lr_scheduler.OneCycleLR, 
                                           max_lr=1e-2, 
                                           total_steps=finetune_epochs)


