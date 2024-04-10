#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import grad 

import pytorch_lightning as pl
from torchdyn.numerics.odeint import odeint_hybrid
from torchdyn.numerics.solvers import DormandPrince45
import torchdyn.numerics.sensitivity
import attr

## lietorch:
import sys; sys.path.append('../')
import lie_torch as lie
import math

from functorch import vmap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# <h1> Quadratic Shaping Only </h1>

# Class Definition: Mixed Dynamics on SE3 with NNs for potential and damping injection

# In[2]:


from models import SE3_Dynamics, SE3_Quadratic, AugmentedSE3, PosDefSym, PosDefTriv, PosDefSym_Small
        


# Sensitivity / Adjoint Method: v1
# 
# This only needs the final condition of the forward dynamics and adjoint dynamics, then it computes both state and adjoint state from there

# In[3]:


from sensitivity import _gather_odefunc_hybrid_adjoint_light


# Learners: Training step, etc.

# In[4]:


from learners import EnergyShapingLearner


# Event definition: Chart Switching on SE(3)

# In[5]:


# Adapted from pytorch-implicit/exampels/network/simulate_tcp.ipynb, and Paper: Neural Hybrid Automata: Learning Dynamics withMultiple Modes and Stochastic Transitions (M.Poli, 2021)

@attr.s
class EventCallback(nn.Module):
    def __attrs_post_init__(self):
        super().__init__()

    def check_event(self, t, x):
        raise NotImplementedError

    def jump_map(self, t, x):
        raise NotImplementedError
    
    def batch_jump(self, t, x, ev):
        raise NotImplementedError

@attr.s
class ChartSwitch(EventCallback):       
    def check_event(self, t, xi): 
        # works for collection of states
        qi, P, i, rem = xi[...,:6], xi[...,6:12], xi[...,12], xi[...,13:]
        wi = qi[...,:3]
        ev = (torch.sqrt(lie.dot(wi,wi)) > math.pi*3/4).bool() 
        return ev

    def jump_map(self, t, xi):
        #xi = torch.squeeze(xi)
        qi, P, i, rem = xi[...,:6], xi[...,6:12], xi[...,12], xi[...,13:]
        H = lie.unchart(qi, i)
        j = torch.unsqueeze(lie.bestChart(H),0)
        qj = lie.chart_trans(qi, i, j)
        return torch.cat((qj, P,j,rem), -1) #torch.unsqueeze(,0)
    
    def batch_jump(self, t, xi, ev):
        xi[ev,:] = vmap(self.jump_map)(t[ev],xi[ev,:])
        return xi
        
    
@attr.s
class ChartSwitchAugmented(EventCallback):
    des_props = None
    # Expects x of type: z[:6] = qi, z[6:12] = P, z[12] = i, z[13:25] = λi, z[25:] = μ. This is used for the system augmented with co-state-dynamics for adjoint gradient method
    def check_event(self, t, z): 
        xi, i, λi, rem = self.to_input(z)
        w = xi[...,:3]
        ev = (torch.sqrt(lie.dot(w,w)) > math.pi*3/4).bool()  
        return ev

    def jump_map(self, t, z):
        xi, i, λi, rem  = z[...,:12], z[...,12], z[...,13:25], z[...,25:]
        qi = xi[...,:6]
        H = lie.unchart(qi, i)
        j = lie.bestChart(H)
        xj = lie.chart_trans_mix(xi, i, j)
        λj = lie.chart_trans_mix_Co(xi, λi, i, j)
        return torch.cat((xj, torch.unsqueeze(j,-1), λj, rem), -1)
    
    def batch_jump(self, t, z, ev):
        xi, i, λii, rem = self.to_input(z)
        z = torch.cat((xi,torch.unsqueeze(i,-1),λii),-1)
        z[ev,:] = vmap(self.jump_map)(t[:xi.shape[0]][ev],z[ev,:])
        xi, i, λii = z[...,:12], z[...,12], z[...,13:26] 
        return self.to_output(xi, i, λii, rem)
    
    def to_input(self, z):
        if (self.des_props!=None):
            numels,shapes = tuple(self.des_props)
            xii_nel, λi_nel = tuple(numels)
            xii_shp, λi_shp = tuple(shapes)
            xii, λii, rem = z[:xii_nel], z[xii_nel:xii_nel+λi_nel], z[xii_nel+λi_nel:]
            xii, λii = xii.reshape(xii_shp), λii.reshape(λi_shp)
            xi, i = xii[...,:12], torch.unsqueeze(xii[...,12],-1)
            return xi, i, λii, rem
        else:
            xi, i, λi, rem  = z[...,:12], z[...,12], z[...,13:25], z[...,25:]
            return xi, i, λi, rem
    
    def to_output(self, xj, j, λj, rem):
        if (self.des_props!= None):
            xjj = torch.cat((xj,torch.unsqueeze(j,-1)),-1)
            z = torch.cat((xjj.flatten(),λj.flatten(),rem))
        else:
            z = torch.cat((xj, torch.unsqueeze(j,-1), λj, rem), -1)
        return z


# Parameters of Dynamics, Definition of Loss-Function

# In[6]:


# Adapted from latent-energy-shaping-main/notebooks/optimal_energy_shaping.ipynb

I = torch.diag(torch.tensor((0.01,0.01,0.01,1,1,1))).to(device) ; # Inertia Tensor

from models import IntegralLoss_Quadratic

    


# Definition of NNs for potential and damping injection

# In[7]:


### 4 components of potential function, one per chart:
# nh = 32

Kd = nn.Sequential(nn.Linear(1, 6),PosDefTriv()) #nn.Sequential(nn.Linear(1, nh),nn.Softplus(), nn.Linear(nh,nh), nn.Softplus(), nn.Linear(nh, 21),PosDefSym())
G0 = nn.Sequential(nn.Linear(1, 3),PosDefTriv()) #nn.Sequential(nn.Linear(1, nh),nn.Softplus(), nn.Linear(nh,nh), nn.Softplus(), nn.Linear(nh,6),PosDefSym_Small())
Gt = nn.Sequential(nn.Linear(1, 3),PosDefTriv()) #nn.Sequential(nn.Linear(1, nh),nn.Softplus(), nn.Linear(nh,nh), nn.Softplus(), nn.Linear(nh,6),PosDefSym_Small())

### Initialize Parameters: 

for net in (Kd,G0,Gt):
    for p in net.parameters():torch.nn.init.zeros_(p)
        


# Initialization of Chart-Switches, Prior- \& Target Distribution, and Dynamics

# In[8]:



### Prior and Target Distribution

from utils import prior_dist_SE3, target_dist_SE3, multinormal_target_dist

th_max = torch.tensor(math.pi).to(device); d_max = torch.tensor(1).to(device); pw_max = torch.tensor(0.03).to(device); pv_max = torch.tensor(1).to(device); ch_min = torch.tensor(0).to(device); ch_max = torch.tensor(0).to(device); 
prior = prior_dist_SE3(th_max,d_max,pw_max,pv_max,ch_min,ch_max,device)

H_target = torch.eye(4).to(device); sigma_th = torch.tensor(1).to(device); sigma_d = torch.tensor(1).to(device); sigma_pw = torch.tensor(1e-1).to(device); sigma_p = torch.tensor(1e-1).to(device);
target = target_dist_SE3(H_target,sigma_th,sigma_d,sigma_pw,sigma_p,device)

### Callback:
callbacks = [ChartSwitch()]
jspan = 10 # maximum number of chart switches per iteration (if this many happen, something is wrong anyhow)

callbacks_adjoint = [ChartSwitchAugmented()]
jspan_adjoint = 10

### Initialize Dynamics of Quadratic Controller

t_span = torch.linspace(0, 1, 30).to(device) 

f_Quadratic = SE3_Quadratic(I,Kd,G0,Gt,H_target).to(device) # torch.load('f_Quadratic.pt')

aug_f_Quadratic = AugmentedSE3(f_Quadratic, IntegralLoss_Quadratic(f_Quadratic)).to(device) 


# Trainign Loop: Quadratic Potential

# In[9]:


from HybridODE import NeuralODE_Hybrid
from pytorch_lightning.loggers import WandbLogger

solver = 'dopri5'
atol, rtol, atol_adjoint, rtol_adjoint = 1e-4,1e-4,1e-4,1e-4
dt_min, dt_min_adjoint = 0, 0

model_Q = NeuralODE_Hybrid(f_Quadratic, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, dt_min, atol_adjoint, rtol_adjoint, dt_min_adjoint, IntegralLoss_Quadratic(f_Quadratic), sensitivity = 'hybrid_adjoint_full').to(device) 
aug_model_Q = NeuralODE_Hybrid(aug_f_Quadratic, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, dt_min, atol_adjoint, rtol_adjoint, dt_min_adjoint, sensitivity = 'hybrid_adjoint_full').to(device) 
#model_Q = NeuralODE_Hybrid(f_Quadratic, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, atol_adjoint, rtol_adjoint, IntegralLoss_Quadratic(f_Quadratic)).to(device) 
#aug_model_Q = NeuralODE_Hybrid(aug_f_Quadratic, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, atol_adjoint, rtol_adjoint).to(device) 
learn_Q = EnergyShapingLearner(model_Q, t_span, prior, target, aug_model_Q).to(device) 
learn_Q.lr = 1e-2
learn_Q.batch_size = 100
logger_Q = WandbLogger(project='quadratic-shaping-SE3',name='quadratic-controller')
trainer = pl.Trainer(max_epochs=1000, logger=logger_Q,gpus = torch.cuda.device_count())#
trainer.fit(learn_Q)
torch.save(f_Quadratic,'f_Quadratic.pt')


# Training Loop: Optimal Potential Shaping

# Evaluation: Plots and stuff

# In[ ]:





# In[ ]:





# In[ ]:




