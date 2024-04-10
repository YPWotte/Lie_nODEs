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


# <h1> Only General Potential Shaping: gradients immediately on algebra, potential on SE(3), full-state integral loss, gravity</h1>

# Class Definition: Mixed Dynamics on SE3 with NNs for potential and damping injection

# In[2]:


from models import se3_Dynamics,SE3_Dynamics, SE3_Quadratic, AugmentedSE3, PosDefSym, PosDefTriv, PosDefSym_Small


# Sensitivity / Adjoint Method:
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
        
    def jump_map_forced(self, t, xi, j):
        # jump to chart j for all
        #xi = torch.squeeze(xi)
        qi, P, i, rem = xi[...,:6], xi[...,6:12], xi[...,12], xi[...,13:]
        qj = qi*0;
        for k in range(qi.size(0)):
            qj[k,:] = lie.chart_trans(qi[k,:], i[k], j[k])
        return torch.cat((qj, P,torch.unsqueeze(j,1),rem), -1) #torch.unsqueeze(,0)
    
    def batch_jump_forced(self, t, xi, j):
        xi = vmap(self.jump_map_forced)(t,xi,j)
        return xi
        
    
@attr.s
class ChartSwitchAugmented(EventCallback):
    des_props = None
    # Expects x of type: z[:6] = qi, z[6:12] = P, z[12] = i, z[13:25] = λi, z[25:] = μ. This is used for the system augmented with co-state-dynamics for adjoint gradient method
    def check_event(self, t, z): 
        xi, i, λi, rem = self.to_input(z)
        w = xi[...,:3]
        ev = (torch.sqrt(lie.dot(w,w)) < -1).bool()  
        return ev

    def batch_jump_to_Id(self, λiT, xT):
        #only transitions λiT to corresponding version λT at identity of SE(3)
        λT = vmap(lie.dchart_trans_mix_Co_to_Id)(λiT,xT[...,:12],xT[...,-1])
        return λT
    
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

from models import IntegralLoss_Full


# Definition of NNs for potential and damping injection

# In[7]:


nh = 32

V = nn.Sequential(nn.Linear(12, nh), nn.Softplus(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, 1)).to(device)
#V = nn.Sequential(nn.Linear(12, nh), nn.Softplus(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, 1)).to(device)

### Likewise for Damping injection:
nf = 6
B = nn.Sequential(nn.Linear(18, nh), nn.Softplus(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, nf),PosDefTriv()).to(device)
#B = nn.Sequential(nn.Linear(18, nh), nn.Softplus(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, nf),PosDefTriv()).to(device)

### Initialize Parameters: 

for p in V.parameters(): torch.nn.init.normal_(p, mean=0.0, std=0.01)#torch.nn.init.zeros_(p)
for p in B.parameters(): torch.nn.init.normal_(p, mean=0.0, std=0.01) #torch.nn.init.zeros_(p)


# Initialization of Chart-Switches, Prior- \& Target Distribution, and Dynamics

# In[8]:


### Prior and Target Distribution

from utils import prior_dist_SE3, target_dist_SE3_HP,  target_cost_SE3, multinormal_target_dist

th_max = torch.tensor(math.pi).to(device); d_max = torch.tensor(1).to(device); pw_max = torch.tensor(0.001).to(device); pv_max = torch.tensor(0.1).to(device); ch_min = torch.tensor(0).to(device); ch_max = torch.tensor(0).to(device); 
prior = prior_dist_SE3(th_max,d_max,pw_max,pv_max,ch_min,ch_max,device)

H_target = torch.eye(4).to(device); sigma_th = torch.tensor(4).to(device); sigma_d = torch.tensor(4).to(device); sigma_pw = torch.tensor(5).to(device); sigma_p = torch.tensor(5e-4).to(device);
                                    # sigma_th = torch.tensor(0.4).to(device); sigma_d = torch.tensor(0.4).to(device); sigma_pw = torch.tensor(1e-1).to(device); sigma_p = torch.tensor(1e-1).to(device);
H_target[2,3] = -1;
    
target = target_cost_SE3(H_target,sigma_th,sigma_d,sigma_pw,sigma_p,device)

# Integral Loss
i_su = torch.tensor(1e-1).to(device)
#i_suw = torch.tensor(100).to(device)
#i_suv = torch.tensor(1e-2).to(device)
i_sw = torch.tensor(1).to(device)
i_sd = torch.tensor(1).to(device)
i_sPw = torch.tensor(1).to(device)
i_sPv = torch.tensor(1e-4).to(device)
i_g = 0*torch.tensor(9.81).to(device)*I[5,5]


### Callback:
callbacks = [ChartSwitch()]
jspan = 10 # maximum number of chart switches per iteration (if this many happen, something is wrong anyhow)

callbacks_adjoint = [ChartSwitchAugmented()]
jspan_adjoint = 10

### Initialize Dynamics 
#(I,B,V) = torch.load('IBV_fShaping_04_08_10:57.pt') #torch.load('IBV_fShaping_21_07_10:07.pt') 
#I = I.to(device); B = B.to(device); V = V.to(device)
f = se3_Dynamics(I,B,V,target).to(device) # (I,B,V) = torch.load('IBV_fShaping.pt')

## Augmented Dynamics with integral loss
aug_f = AugmentedSE3(f, IntegralLoss_Full(f,i_su,i_sw,i_sd,i_sPw,i_sPv,i_g,H_target),target).to(device) 
#aug_f.l.scale = integral_loss_scale


t_span = torch.linspace(0, 3, 30).to(device)  


# Training Loop: Optimal Potential Shaping

# In[9]:


#f.parameters()
#model_parameters = filter(lambda p: p.requires_grad, f.parameters())
#params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
#print(params)
#model_parameters = filter(lambda p: p.requires_grad, f.parameters())

#[print(p) for p in model_parameters]


# In[10]:


from HybridODE import NeuralODE_Hybrid
from pytorch_lightning.loggers import WandbLogger
import datetime

today = datetime. datetime. now()
date_time = today. strftime("%d/%m_%H:%M")

solver = 'dopri5'
atol, rtol, atol_adjoint, rtol_adjoint = 1e-4,1e-5,1e-4,1e-5
dt_min, dt_min_adjoint = 0, 0

model = NeuralODE_Hybrid(f, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, dt_min, atol_adjoint, rtol_adjoint, dt_min_adjoint, IntegralLoss_Full(f,i_su,i_sw,i_sd,i_sPw,i_sPv,i_g,H_target), sensitivity = 'hybrid_adjoint_full').to(device) 
aug_model = NeuralODE_Hybrid(aug_f, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, dt_min, atol_adjoint, rtol_adjoint, dt_min_adjoint, sensitivity = 'hybrid_adjoint_full').to(device) 
#model.integral_loss.scale = integral_loss_scale

learn = EnergyShapingLearner(model, t_span, prior, target, aug_model).to(device) 

learn.lr = 1e-3
learn.batch_size = 2048

logger = WandbLogger(project='potential-shaping-SE3', name='NN_'+date_time)

trainer = pl.Trainer(max_epochs=1, logger=logger, gpus = torch.cuda.device_count())#
trainer.fit(learn)
#torch.save((I,B,V),'IBV_fShaping_'+date_time+'.pt')


# In[11]:


date_time = today.strftime("%d_%m_%H:%M")
torch.save((I,B,V),'IBV_fShaping_A_'+date_time+'.pt')

#%debug


# In[12]:


#%debug


# In[ ]:





# In[ ]:




