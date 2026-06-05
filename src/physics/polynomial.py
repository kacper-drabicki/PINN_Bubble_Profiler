import torch
from dataclasses import dataclass
from functools import partial
   
    
class Potential:
    
    def __init__(self, c: float):
        self.c = c

    def V(self, y):
        return (-4*self.c+3)/2 * y**2 - y**3 + self.c * y**4

    def dV(self, y):
        return (-4*self.c+3) * y - 3*y**2 + 4* self.c * y**3

class Loss:
    
    def __init__(self, potential: Potential, is_pretrain: bool = True):
        self.potential = potential
        self.is_pretrain = is_pretrain

    def __call__(self, r: torch.Tensor, y: torch.Tensor):
        phi_r = torch.autograd.grad(y, r, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        phi_rr = torch.autograd.grad(phi_r, r, grad_outputs=torch.ones_like(phi_r), create_graph=True)[0]

        loss_physics = torch.mean((phi_rr + 2 / (r) * phi_r - self.potential.dV(y))**2)
        if self.is_pretrain:
            loss_boundary_zero = torch.mean((y[0:1]-1.0)**2)
        else:
            loss_boundary_zero = torch.mean(phi_r[0:1]**2)
        loss_boundary_max = torch.mean((y[-1:])**2)

        return loss_physics + loss_boundary_zero + loss_boundary_max 
    
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    modelPath: str = "saved_models/basic_model.pth" 
    output_dim: int = 1
    # fizyka
    c: float = 0.51
    r_max: float = 55.0
    # pretrain
    pretrain_epochs: int = 50000
    pretrain_loss_fn: callable = Loss(Potential(c), is_pretrain=True)
    pretrain_optimizer: callable = partial(torch.optim.Adam, lr=1e-2)
    pretrain_scheduler: callable = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, 
                                           mode='min', 
                                           factor=0.5,
                                           patience=100,
                                           threshold=1e-6)
    # finetune
    finetune_epochs: int = 200000
    finetune_loss_fn: callable = Loss(Potential(c), is_pretrain=False)
    finetune_optimizer: callable = partial(torch.optim.Adam, lr=1e-2)
    finetune_scheduler: callable = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, 
                                           mode='min', 
                                           factor=0.1,
                                           patience=100,
                                           threshold=1e-6)

