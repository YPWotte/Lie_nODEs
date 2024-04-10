#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:19:34 2023

@author: yannik
"""


###############
#### Setup
###############

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

from models import se3_Dynamics,SE3_Dynamics, SE3_Quadratic, AugmentedSE3, PosDefSym, PosDefTriv, PosDefSym_Small
from sensitivity import _gather_odefunc_hybrid_adjoint_light
from learners import EnergyShapingLearner
from models import IntegralLoss_Full
from utils import prior_dist_SE3, target_dist_SE3_HP,  target_cost_SE3, multinormal_target_dist
from HybridODE import NeuralODE_Hybrid
from pytorch_lightning.loggers import WandbLogger
import datetime

## Making folders
import os

## Plotting Control Performance:
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mpl.rcParams.update(mpl.rcParamsDefault)
# matplotlib.use("xxx") # Agg for default, SVG or Cairo for SVG


## Plotting Training Progress:
import csv
import numpy as np
import pandas as pd

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



###############
#### Loop over various training outcomes
###############


FOLDER = []
NAME_IBV = ['quadratic_26_07_10:33', 'A_01_09_16:46','A_18_08_12:09','04_08_10:57','21_07_10:07']
H_Target_23 = [0,-1,-2,0,0]
DUPPER = [1,3,3,1,1]

for (name,H_23,d_upper) in zip(NAME_IBV,H_Target_23,DUPPER):
    if not os.path.exists('./Results/Figures/'+name):
        os.makedirs('./Results/Figures/'+name)
    ###############
    #### Initial definitions
    ###############
    I = torch.diag(torch.tensor((0.01,0.01,0.01,1,1,1))).to(device) ; # Inertia Tensor
    
    
    nh = 32
    
    V = nn.Sequential(nn.Linear(12, nh), nn.Softplus(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, 1)).to(device)
    
    nf = 6
    B = nn.Sequential(nn.Linear(18, nh), nn.Softplus(), nn.Linear(nh, nh), nn.Tanh(), nn.Linear(nh, nf),PosDefTriv()).to(device)
    
    for p in V.parameters(): torch.nn.init.normal_(p, mean=0.0, std=0.01)#torch.nn.init.zeros_(p)
    for p in B.parameters(): torch.nn.init.normal_(p, mean=0.0, std=0.01) #torch.nn.init.zeros_(p)
    
    
    
    th_max = torch.tensor(math.pi).to(device); d_max = torch.tensor(1).to(device); pw_max = torch.tensor(0.001).to(device); pv_max = torch.tensor(0.1).to(device); ch_min = torch.tensor(0).to(device); ch_max = torch.tensor(0).to(device); 
    prior = prior_dist_SE3(th_max,d_max,pw_max,pv_max,ch_min,ch_max,device)
    
    H_target = torch.eye(4).to(device); sigma_th = torch.tensor(4).to(device); sigma_d = torch.tensor(10).to(device); sigma_pw = torch.tensor(5).to(device); sigma_p = torch.tensor(1).to(device);
                                        # sigma_th = torch.tensor(0.4).to(device); sigma_d = torch.tensor(0.4).to(device); sigma_pw = torch.tensor(1e-1).to(device); sigma_p = torch.tensor(1e-1).to(device);
    H_target[2,3] = H_23;
    
    target = target_cost_SE3(H_target,sigma_th,sigma_d,sigma_pw,sigma_p,device)
    
    i_su = torch.tensor(1).to(device)
    i_sw = torch.tensor(1).to(device)
    i_sd = torch.tensor(1).to(device)
    i_sPw = torch.tensor(1).to(device)
    i_sPv = torch.tensor(1).to(device)
    
    callbacks = [ChartSwitch()]
    jspan = 10 # maximum number of chart switches per iteration (if this many happen, something is wrong anyhow)
    
    callbacks_adjoint = [ChartSwitchAugmented()]
    jspan_adjoint = 10
    
    (I,B,V) = torch.load('IBV_fShaping_'+name+'.pt')# 'IBV_fShaping_quadratic_26_07_10:33.pt' (quadratic) #'IBV_fShaping_A_01_09_16:46.pt' (H_target[2,3] = -1), #torch.load('IBV_fShaping_A_18_08_12:09.pt') (H_target[2,3] = -2), # First gravity potential: torch.load('IBV_fShaping_04_08_10:57.pt') (H_target[2,3] = 0) # Best full potential: 'IBV_fShaping_21_07_10:07.pt'
    I = I.to(device); B = B.to(device); V = V.to(device)
    f = se3_Dynamics(I,B,V,target).to(device) # (I,B,V) = torch.load('IBV_fShaping.pt')
    
    i_g = torch.tensor(9.81*I[5,5])
    aug_f = AugmentedSE3(f, IntegralLoss_Full(f,i_su,i_sw,i_sd,i_sPw,i_sPv,i_g,H_target),target).to(device) 
    
    
    t_span = torch.linspace(0, 3, 30).to(device)  
    
    
    today = datetime. datetime. now()
    date_time = today. strftime("%d/%m_%H:%M")
    
    solver = 'dopri5'
    atol, rtol, atol_adjoint, rtol_adjoint = 1e-5,1e-6,1e-5,1e-6
    dt_min, dt_min_adjoint = 0, 0
    
    model = NeuralODE_Hybrid(f, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, dt_min, atol_adjoint, rtol_adjoint, dt_min_adjoint, IntegralLoss_Full(f,i_su,i_sw,i_sd,i_sPw,i_sPv,i_g,H_target), sensitivity = 'hybrid_adjoint_full').to(device) 
    aug_model = NeuralODE_Hybrid(aug_f, jspan, callbacks, jspan_adjoint, callbacks_adjoint, solver, atol, rtol, dt_min, atol_adjoint, rtol_adjoint, dt_min_adjoint, sensitivity = 'hybrid_adjoint_full').to(device) 
    #model.integral_loss.scale = integral_loss_scale
    
    learn = EnergyShapingLearner(model, t_span, prior, target, aug_model).to(device) 
    
    learn.lr = 1e-3
    learn.batch_size = 2048
    
    
    ##################
    
    batch_size = 100;
    t_span = torch.linspace(0, 21, 1000).to(device);
    
    x0 = prior.sample((batch_size)).to(device)
    t, xT = model.forward(x0,t_span)
    
    
    ###############
    #### Plotting: Characterization of controller
    ###############
    
    plt.style.use("./mystyle2.txt")
    fig1 = plt.figure()#figsize=(10, 10))
    ax1 = plt.subplot()#aspect=1) 
    
    # plt.rcParams.update({"text.usetex": True})
    # plt.style.use("./mystyle.txt")
    fig2 = plt.figure()#figsize=(10, 10))
    ax2 = plt.subplot()#aspect=1) 
    # d_upper = 3
    #plt.rcParams.update({"text.usetex": True})
    
    
    
    ## Loop over S, plot various batches of type X,Y individually
    for i in range(100):
        xiT = xT[:,i,:]
        T = t.detach().numpy()
        th, d = target.distance(xiT)
        th = th.detach().numpy()
        d = d.detach().numpy()
        c = "r"
        lw = 1.0
        ax1.plot(T, th, c=c, alpha=0.25, lw=lw) 
        ax2.plot(T, d, c=c, alpha=0.25, lw=lw)
    
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, np.pi)
    ax1.set_xlabel("Time in s", usetex=True)
    ax1.set_ylabel("Angle in rad", usetex=True)
    ax1.set_title("Angle to goal pose for %d trajectories of rigid bodies" % 100, usetex=True)
    ax1.set_yticks([0, np.pi/2, np.pi])
    ax1.set_yticklabels(["0", "$\pi$/2", "$\pi$"], usetex =True)
    ax1.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax1.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3"], usetex=True)
    plt.tight_layout()
    fig1.savefig('./Results/Figures/'+name+'/Angle_v_Time_3s_'+name+'.svg')
    
    # ax1.plot(1.005, 0, ">k",
    # transform=ax1.get_yaxis_transform(), clip_on=False)
    
    
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, d_upper)
    ax2.set_xlabel("Time in s", usetex=True)
    ax2.set_ylabel("Distance in m", usetex=True)
    ax2.set_title("Distance to goal pose for %d trajectories of rigid bodies" % 100, usetex=True)
    if (d_upper == 3):
        ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3]) 
        ax2.set_yticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3" ], usetex =True)
    else:
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
        ax2.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], usetex =True) 
    ax2.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax2.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3"], usetex=True)
    plt.tight_layout()
    fig2.savefig('./Results/Figures/'+name+'/Dist_v_Time_3s_'+name+'.svg')
    
    # ax2.plot(1.005, 0, ">k",
    # transform=ax2.get_yaxis_transform(), clip_on=False)
    
    
    
    ################
    
    plt.style.use("./mystyle2.txt")
    plt.rcParams.update({"text.usetex": True})
    fig1 = plt.figure()#figsize=(10, 10))
    ax1 = plt.subplot()#aspect=1) 
    
    #plt.rcParams.update({"text.usetex": True})
    # plt.style.use("./mystyle.txt")
    fig2 = plt.figure()#figsize=(10, 10))
    ax2 = plt.subplot()#aspect=1) 
    
    #plt.rcParams.update({"text.usetex": True})
    
    
    
    ## Loop over S, plot various batches of type X,Y individually
    for i in range(100):
        xiT = xT[:,i,:]
        T = t.detach().numpy()
        th, d = target.distance(xiT)
        th = th.detach().numpy()
        d = d.detach().numpy()
        c = "r"
        lw = 1.0
        ax1.plot(T, th, c=c, alpha=0.25, lw=lw) 
        ax2.plot(T, d, c=c, alpha=0.25, lw=lw)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, np.pi)
    ax1.set_xlabel("Time in s")
    ax1.set_ylabel("Angle in rad")
    ax1.set_title("Angle to goal pose for %d trajectories of rigid bodies" % 100)
    ax1.set_yticks([0, np.pi/2, np.pi])
    ax1.set_yticklabels(["0", "$\pi$/2", "$\pi$"])
    ax1.set_xticks([0, 1, 2, 3, 4 , 5])
    ax1.set_xticklabels(["0", "1", "2", "3", "4", "5"])
    plt.tight_layout()
    fig1.savefig('./Results/Figures/'+name+'/Angle_v_Time_%ds_'%5+name+'.svg')
    
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, d_upper)
    ax2.set_xlabel("Time in s")
    ax2.set_ylabel("Distance in m")
    ax2.set_title("Distance to goal pose for %d trajectories of rigid bodies" % 100)
    if (d_upper == 3):
        ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3]) 
        ax2.set_yticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3" ], usetex =True)
    else:
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
        ax2.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], usetex =True) 
    ax2.set_xticks([0, 1, 2, 3, 4 , 5])
    ax2.set_xticklabels(["0", "1", "2", "3", "4", "5"])
    plt.tight_layout()
    fig2.savefig('./Results/Figures/'+name+'/Dist_v_Time_%ds_'%5+name+'.svg')
    
    ################
    
    
    
    
    
    plt.style.use("./mystyle2.txt")
    fig1 = plt.figure()#figsize=(10, 10))
    ax1 = plt.subplot()#aspect=1) 
    
    #plt.rcParams.update({"text.usetex": True})
    # plt.style.use("./mystyle.txt")
    fig2 = plt.figure()#figsize=(10, 10))
    ax2 = plt.subplot()#aspect=1) 
    
    #plt.rcParams.update({"text.usetex": True})
    
    
    ## Loop over S, plot various batches of type X,Y individually
    for i in range(100):
        xiT = xT[:,i,:]
        T = t.detach().numpy()
        th, d = target.distance(xiT)
        th = th.detach().numpy()
        d = d.detach().numpy()
        c = "r"
        lw = 1.0
        ax1.plot(T, th, c=c, alpha=0.25, lw=lw) 
        ax2.plot(T, d, c=c, alpha=0.25, lw=lw)
    
    ax1.set_xlim(0, 21)
    ax1.set_ylim(0, np.pi)
    ax1.set_xlabel("Time in s", usetex=True)
    ax1.set_ylabel("Angle in rad", usetex=True)
    ax1.set_title("Angle to goal pose for %d trajectories of rigid bodies" % 100, usetex=True)
    ax1.set_yticks([0, np.pi/2, np.pi])
    ax1.set_yticklabels(["0", "$\pi$/2", "$\pi$"], usetex =True)
    ax1.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    ax1.set_xticklabels(["0", "3", "6", "9", "12", "15", "18", "21" ], usetex=True)
    plt.tight_layout()
    fig1.savefig('./Results/Figures/'+name+'/Angle_v_Time_21s_'+name+'.svg')
    
    # ax1.plot(1.005, 0, ">k",
    # transform=ax1.get_yaxis_transform(), clip_on=False)
    
    
    ax2.set_xlim(0, 21)
    ax2.set_ylim(0, d_upper)
    ax2.set_xlabel("Time in s", usetex=True)
    ax2.set_ylabel("Distance in m", usetex=True)
    ax2.set_title("Distance to goal pose for %d trajectories of rigid bodies" % 100, usetex=True)
    if (d_upper == 3):
        ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3]) 
        ax2.set_yticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3" ], usetex =True)
    else:
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
        ax2.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], usetex =True) 
    ax2.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    ax2.set_xticklabels(["0", "3", "6", "9", "12", "15", "18", "21" ], usetex=True)
    plt.tight_layout()
    fig2.savefig('./Results/Figures/'+name+'/Dist_v_Time_21s_'+name+'.svg')
    
    
    #####################
    
    
    
    plt.style.use("./mystyle2.txt")
    plt.rcParams.update({"text.usetex": True})
    
    N = 1000
    w = torch.rand(3)
    v = torch.rand(3);
    
    w = w/lie.angle(w);
    v = v/lie.angle(v);
    #t = torch.cat((w,v))
    V = np.zeros((N));
    th = np.zeros((N));
    wn = torch.rand(1)
    
    ## Distance only, arbitrary angle
    for i in range(N):
        #ti = t*torch.pi/N*i;
        #ti = t*i/N;
        ti = torch.cat((w*wn,v*i/N))
        H = H_target @ lie.exp_SE3(ti);
        V[i] = f.Potential(H)
        #th[i] = lie.angle(ti)
        th[i] = lie.angle(ti[3:])
    
    ylabel = 'Potential'
    xlabel = 'Distance in m'
    title = 'Potential vs. Distance, arbitrary angle'
    fig = plt.figure()
    ax = plt.subplot()
    plt.plot(th,V)
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/Pot_v_Dist_'+name+'.svg')
    
    ## Distance and angle
    for i in range(N):
        #ti = t*torch.pi/N*i;
        ti =  torch.cat((w*torch.pi,v))*i/N;
        #ti = torch.cat((w*wn,v*i/N))
        H = H_target @ lie.exp_SE3(ti);
        V[i] = f.Potential(H)
        #th[i] = lie.angle(ti)
        th[i] = lie.angle(ti[3:])
    
    ylabel = 'Potential'
    xlabel = 'Distance in m'
    title = 'Potential vs. Screw'
    plt.figure()
    ax = plt.subplot()
    plt.plot(th,V)
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/Pot_v_Screw_'+name+'.svg')
    
    ## Angle only, zero distance
    for i in range(N):
        ti = torch.cat((w*2*torch.pi*i/N,v*0))
        H = H_target @ lie.exp_SE3(ti);
        V[i] = f.Potential(H)
        th[i] = lie.angle(ti)
    
    ylabel = 'Potential'
    xlabel = 'Angle in rad'
    title = 'Potential vs. Angle'
    plt.figure()
    ax = plt.subplot()
    plt.plot(th,V)
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/Pot_v_Ang_'+name+'.svg')
    
    
    ## Distance outside training range:
    
    for i in range(N):
        #ti = t*torch.pi/N*i;
        #ti = t*i/N;
        ti = torch.cat((w*0,v*50*i/N))
        H = H_target @ lie.exp_SE3(ti);
        V[i] = f.Potential(H)
        #th[i] = lie.angle(ti)
        th[i] = lie.angle(ti[3:])
    
    ylabel = 'Potential'
    xlabel = 'Distance in m'
    title = 'Potential vs. Distance outside training range'
    plt.figure()
    ax = plt.subplot()
    plt.plot(th,V)
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/Pot_v_Dist_outside_range_'+name+'.svg')
    
    ## Potential: compare potential along various paths for measure of symmetry
    
    ## Distance only, towards increasing height
    x = torch.tensor((2,0,0));
    y = torch.tensor((0,2,0));
    z = torch.tensor((0,0,2));
    V0 = np.zeros((N));
    V1 = np.zeros((N));
    V2 = np.zeros((N))
    for i in range(N):
        #ti = t*torch.pi/N*i;
        #ti = t*i/N;
        ti_0 = torch.cat((w*0,x*(i/N - 1/2)))
        ti_1 = torch.cat((w*0,y*(i/N - 1/2)))
        ti_2 = torch.cat((w*0,z*(i/N - 1/2)))
        H0 = H_target @ lie.exp_SE3(ti_0);
        H1 = H_target @ lie.exp_SE3(ti_1);
        H2 = H_target @ lie.exp_SE3(ti_2);
        V0[i] = f.Potential(H0)
        V1[i] = f.Potential(H1)
        V2[i] = f.Potential(H2)
        #th[i] = lie.angle(ti)
        th[i] = lie.angle(ti_1[3:])*np.sign(i/N-1/2)
    
    ylabel = 'Potential'
    xlabel = 'Distance in m'
    title = 'Potential vs. Distance, different directions'
    fig = plt.figure()
    ax = plt.subplot()
    plt.plot(th,V0)
    plt.plot(th,V1)
    plt.plot(th,V2)
    #ax.set_ylim((-4.5,-3))
    ax.set_xlim((-1,1))
    plt.legend(('Potential vs. x','Potential vs. y', 'Potential vs. z'))
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/3_Pot_v_Dist_'+name+'.svg')
    
    
    
    
    ## Damping: magnitude vs time, 
    #   magnitude vs. angle, magnitude vs. distance: in both cases, average over various samplings of momentum
    #   compare magnitude of damping along various paths for measure of symmetry
    #   magnitude of angular part vs. magnitude of linear part
    #   check off-center elements of damping matrix
    # -> These are too complicated
    
    ## Damping is largely characterized by the extent to which it dissipates energy, and the extent to which it re-routes kinetic energy
    # Interesting plots are:
    # - cumulative dissipated energy -> plot of initial -(kinetic + potential) energy 
    # kinetic + potential energy
    # plot of various kinetic energy components, albeit far more difficult to characterize 
    
    
    ## Energy components
    ylabel = 'Energy'
    xlabel = 'Time in s'
    title = 'Various components of energy vs. time'
    plt.figure()
    ax = plt.subplot()
    
    
    for i in range(100):
        xiT = xT[:,i,:]
        T = t.detach().numpy()
        ETot, EPot, EKin, EKinRot, EKinTrans = vmap(f.Energy)(xiT)
        ETot = ETot.detach().numpy(); EPot = EPot.detach().numpy(); EKin = EKin.detach().numpy(); EKinRot = EKinRot.detach().numpy(); EKinTrans = EKinTrans.detach().numpy()
        EPot = EPot - EPot[-1]
        ETot = EPot[0]+EKin[0]+ ETot*0
        
        EDamp = ETot[0] - EPot - EKin
        cTot = "#FCC419"#"y"
        cPot = "#2F9E44"#"g"
        cKin = "#4C6EF5"#"b"
        cDamp = "#FA5252" #"r"
        lw = 1.0
        ax.plot(T, ETot, c=cTot, alpha=0.25, lw=lw) 
        ax.plot(T, EPot, c=cPot, alpha=0.25, lw=lw) 
        ax.plot(T, EKin, c=cKin, alpha=0.25, lw=lw) 
        ax.plot(T, EDamp, c=cDamp, alpha=0.25, lw=lw) 
    
    plt.legend(('Total Energy','Potential Energy', 'Kinetic Energy', 'Dissipated Energy'))
    ax.set_xlim(0, 3)
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/Energy_components_'+name+'.svg')
    
    ## Energy components, scaled plot
    ylabel = 'Energy'
    xlabel = 'Time in s'
    title = 'Various components of energy vs. time'
    plt.figure()
    ax = plt.subplot()
    
    
    for i in range(100):
        xiT = xT[:,i,:]
        T = t.detach().numpy()
        ETot, EPot, EKin, EKinRot, EKinTrans = vmap(f.Energy)(xiT)
        ETot = ETot.detach().numpy(); EPot = EPot.detach().numpy(); EKin = EKin.detach().numpy(); EKinRot = EKinRot.detach().numpy(); EKinTrans = EKinTrans.detach().numpy()
        EPot = EPot - EPot[-1]
        ETot = EPot[0]+EKin[0]+ ETot*0
        EDamp = ETot[0] - EPot - EKin
        
        EPot = EPot/ETot[0];
        EKin = EKin/ETot[0];
        EDamp = EDamp/ETot[0];
        
        cPot = "g"
        cKin = "b"
        cDamp = "r"
        lw = 1.0
        ax.plot(T, EPot, c=cPot, alpha=0.25, lw=lw) 
        ax.plot(T, EKin, c=cKin, alpha=0.25, lw=lw) 
        ax.plot(T, EDamp, c=cDamp, alpha=0.25, lw=lw) 
    
    plt.legend(('Potential Energy', 'Kinetic Energy', 'Dissipated Energy'))
    ax.set_xlim(0, 3)
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/Energy_components_normalized_'+name+'.svg')
    
    
    ## Energy components, global average
    ylabel = 'Energy'
    xlabel = 'Time in s'
    title = 'Various components of energy vs. time'
    plt.figure()
    ax = plt.subplot()
    
    xiT = xT[:,0,:]
    ETot, EPot, EKin, EKinRot, EKinTrans = vmap(f.Energy)(xiT)
    ETotPrev = ETot*0
    EPotPrev = EPot*0
    EKinPrev = EKin*0
    EDampPrev = EDamp*0 
    EKinTransPrev = EKinTrans*0
    EKinRotPrev = EKinRot*0
    
    for i in range(100):
        xiT = xT[:,i,:]
        T = t.detach().numpy()
        ETot, EPot, EKin, EKinRot, EKinTrans = vmap(f.Energy)(xiT)
        ETot = ETot+ETotPrev; ETotPrev = ETot 
        EPot = EPot+EPotPrev; EPotPrev = EPot 
        EKin = EKin+EKinPrev; EKinPrev = EKin 
        EDamp = EDamp+EDampPrev; EDampPrev = EDamp 
        EKinTrans = EKinTrans+EKinTransPrev; EKinTransPrev = EKinTrans 
        EKinRot = EKinRot + EKinRotPrev; EKinRotPrev = EKinRot 
    
    ETot = ETot.detach().numpy(); EPot = EPot.detach().numpy(); EKin = EKin.detach().numpy(); EKinRot = EKinRot.detach().numpy(); EKinTrans = EKinTrans.detach().numpy()
    EPot = EPot - EPot[-1]
    ETot = EPot[0]+EKin[0]+ ETot*0
    EDamp = ETot[0] - EPot - EKin
    
    
    
    EPot = EPot/ETot[0];
    EKin = EKin/ETot[0];
    EKinRot = EKinRot/ETot[0];
    EKinTrans = EKinTrans/ETot[0];
    EDamp = EDamp/ETot[0];
        
    EKinTrans_line = EKinTrans;
    EKinRot_line = EKinRot+EKinTrans;
    EDamp_line = EDamp+EKin
    #EKin_line = EKin;
    EPot_line = EPot+EKin+EDamp;
    
    cPot = "#fdae61"#"#2B8A3E"#"g"
    cKinRot ="#abd9e9" #"#862E9C" #"b"
    cKinTrans ="#2c7bb6" #"#15AABF"#"y"
    cDamp = "#d7191c"#"#C92A2A" #"r"
    lw = 0
    ax.plot(T, EPot_line, c=cPot, alpha=1, lw=lw) 
    #ax.plot(T, EKin_line, c=cKin, alpha=0.25, lw=lw) 
    ax.plot(T, EKinTrans_line, c=cKinTrans, alpha=1, lw=lw) 
    ax.plot(T, EKinRot_line, c=cKinRot, alpha=1, lw=lw) 
    ax.plot(T, EDamp_line, c=cDamp, alpha=1, lw=lw) 
    
    ax.fill_between(T,EKinTrans_line,EKinTrans_line*0,facecolor=cKinTrans)
    ax.fill_between(T,EKinRot_line,EKinTrans_line,facecolor=cKinRot)
    ax.fill_between(T,EDamp_line,EKinRot_line,facecolor=cDamp)
    ax.fill_between(T,EPot_line,EDamp_line,facecolor=cPot)
        
    
    leg  = plt.legend(('Potential Energy', 'Kinetic Translational Energy','Kinetic Angular Energy', 'Dissipated Energy'))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    ax.set_xlim(0, 3)
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+name+'/Energy_components_average_cumulative_'+name+'.svg')



###############
#### Plotting: Training Progress
###############

# Comment: Ctrl-4, Uncomment: Ctrl-5


FOLDER = ['General_NN_20-07_to_NN_01-09', 'Quadratic_NN_25-07_to_NN_27-07']
#folder = FOLDER[1]
NAME1 = ['NN_01-09_16:46', 'NN_04-08_10:57', 'NN_07-08_17:16', 'NN_18-08_12:09']#, 'NN_20-07_16:50', 'NN_21-07_10:07']
NAME2 = ['NN_27-07_11:03']
NAMENAME = [NAME1,NAME2]
STORENAME1 = ['A_01_09_16:46','04_08_10:57','NN_07-08_17:16', 'NN_18-08_12:09']
STORENAME2 = ['NN_27-07_11:03']
STORENAME = [STORENAME1,STORENAME2]
for (folder,NAME,STORE) in zip(FOLDER,NAMENAME,STORENAME):
    for (name,store) in zip(NAME,STORE):
        if not os.path.exists('./Results/Figures/'+store):
            os.makedirs('./Results/Figures/'+store)
        ## Read data from csv: Loss, Terminal Loss, Integral Loss
        file = pd.read_csv('./Results/Training Progress/'+folder+'/'+ name+'_All.csv')
        
        T_Full = file['Step'].to_numpy()
        
        T_loss = file['loss'].dropna().index.to_numpy()
        loss = file['loss'].dropna().to_numpy()
        
        T_integral_loss = file['integral_loss'].dropna().index.to_numpy()
        integral_loss = file['integral_loss'].dropna().to_numpy()
        
        T_terminal_loss = file['terminal loss'].dropna().index.to_numpy()
        terminal_loss = file['terminal loss'].dropna().to_numpy()
        
        
        ## Read data from csv: Final state
        T_State = file['th_final'].dropna().index.to_numpy()
        angle = file['th_final'].dropna().to_numpy()
        distance = file['d_final'].dropna().to_numpy()
        Pw = file['Pw_Final'].dropna().to_numpy()
        Pv = file['Pv_Final'].dropna().to_numpy()
        
        
        #pd.merge(file['terminal loss'].dropna(),file['loss'].dropna(),'inner')
        
        #ev = np.isin(T_terminal_loss,T_loss)
        #terminal_loss_small = terminal_loss[ev]
        
        
        ## plot loss data
        
        plt.style.use("./mystyle2.txt")
        plt.rcParams.update({"text.usetex": True})
        
        ylabel = 'Value'
        xlabel = 'Training epoch'
        title = 'Training loss vs. training epoch'
        fig = plt.figure()
        ax = plt.subplot()
        plt.semilogy(T_loss,loss,c = '#7b3294')
        ax.fill_between(T_loss,loss - integral_loss,(loss - integral_loss)*0,facecolor='#008837')
        ax.fill_between(T_loss,loss,loss - integral_loss,facecolor='#c2a5cf')
        #plt.plot(T_integral_loss,integral_loss)
        #plt.plot(T_terminal_loss,terminal_loss)
        #plt.plot(T_loss, loss - integral_loss)
        
        #ax.set_ylim((-4.5,-3))
        #ax.set_xlim((-1,1))
        plt.legend(('Cost $C$', 'Terminal cost $F$','Running cost $r$'))
        ax.set_xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('./Results/Figures/'+store+'/Loss_'+name+'.svg')
        
        
        ## plot state data: angle
        plt.style.use("./mystyle2.txt")
        plt.rcParams.update({"text.usetex": True})
        
        ylabel = 'Angle in rad'
        xlabel = 'Training epoch'
        title = 'Average final angle to goal pose'
        fig = plt.figure()
        ax = plt.subplot()
        plt.semilogy(T_State,angle)
        
        #ax.set_ylim((-4.5,-3))
        #ax.set_xlim((-1,1))
        ax.set_xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('./Results/Figures/'+store+'/Angle_'+name+'.svg')
        
        
        ## plot state data: distance
        plt.style.use("./mystyle2.txt")
        plt.rcParams.update({"text.usetex": True})
        
        ylabel = 'Distance in m'
        xlabel = 'Training epoch'
        title = 'Average final distance to goal pose'
        fig = plt.figure()
        ax = plt.subplot()
        plt.plot(T_State,distance)
        
        #ax.set_ylim((-4.5,-3))
        #ax.set_xlim((-1,1))
        ax.set_xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('./Results/Figures/'+store+'/Distance_'+name+'.svg')
        
        
        ## plot state data: angular momentum
        plt.style.use("./mystyle2.txt")
        plt.rcParams.update({"text.usetex": True})
        
        ylabel = 'Angular momentum in $kg m^2 rad/s$ '
        xlabel = 'Training epoch'
        title = 'Average final angular momentum'
        fig = plt.figure()
        ax = plt.subplot()
        plt.plot(T_State,Pw)
        
        #ax.set_ylim((-4.5,-3))
        #ax.set_xlim((-1,1))
        ax.set_xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('./Results/Figures/'+store+'/Pw_'+name+'.svg')
        
        
        
        ## plot state data
        plt.style.use("./mystyle2.txt")
        plt.rcParams.update({"text.usetex": True})
        
        ylabel = 'Linear momentum in $kg m/s$'
        xlabel = 'Training epoch'
        title = 'Average final linear momentum to goal pose'
        fig = plt.figure()
        ax = plt.subplot()
        plt.plot(T_State,Pv)
        
        #ax.set_ylim((-4.5,-3))
        #ax.set_xlim((-1,1))
        ax.set_xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('./Results/Figures/'+store+'/Pv_'+name+'.svg')



##### Concatenate runs, e.g., for 'NN_20-07_16:50' and  'NN_21-07_10:07'

FOLDER = ['General_NN_20-07_to_NN_01-09', 'Quadratic_NN_25-07_to_NN_27-07']
NAME1 = ['NN_20-07_16:50','NN_25-07_13:08'] #;#
NAME2 = ['NN_21-07_10:07','NN_26-07_10:33'] #;#
STORE = ['21_07_10:07', 'quadratic_26_07_10:33']

for name1,name2,folder,store in zip (NAME1,NAME2,FOLDER,STORE):
    if not os.path.exists('./Results/Figures/'+name1):
        os.makedirs('./Results/Figures/'+name1)
    ## Read data from csv: loss
    file1 = pd.read_csv('./Results/Training Progress/'+folder+'/'+name1+'_All.csv')
    file2 = pd.read_csv('./Results/Training Progress/'+folder+'/'+name2+'_All.csv')
    
    T_Full1 = file1['Step'].to_numpy()
    
    T_loss1 = file1['loss'].dropna().index.to_numpy()
    loss1 = file1['loss'].dropna().to_numpy()
    
    T_integral_loss1 = file1['integral_loss'].dropna().index.to_numpy()
    integral_loss1 = file1['integral_loss'].dropna().to_numpy()
    
    T_terminal_loss1 = file1['terminal loss'].dropna().index.to_numpy()
    terminal_loss1 = file1['terminal loss'].dropna().to_numpy()
    
    T_Full2 = file2['Step'].to_numpy()
    
    T_loss2 = file2['loss'].dropna().index.to_numpy()
    loss2 = file2['loss'].dropna().to_numpy()
    
    T_integral_loss2 = file2['integral_loss'].dropna().index.to_numpy()
    integral_loss2 = file2['integral_loss'].dropna().to_numpy()
    
    T_terminal_loss2 = file2['terminal loss'].dropna().index.to_numpy()
    terminal_loss2 = file2['terminal loss'].dropna().to_numpy()
    
    
    T_Full = np.concatenate((T_Full1,T_Full2+T_Full1[-1]))
    T_loss = np.concatenate((T_loss1,T_loss2+T_Full1[-1]))
    T_integral_loss = np.concatenate((T_integral_loss1,T_integral_loss2+T_Full1[-1]))
    T_terminal_loss = np.concatenate((T_terminal_loss1,T_terminal_loss2+T_Full1[-1]))
    
    loss = np.concatenate((loss1,loss2))
    integral_loss = np.concatenate((integral_loss1,integral_loss2))
    terminal_loss = np.concatenate((terminal_loss1,terminal_loss2))
    
    
    # Read data from csv: state
    
    T_State1 = file1['th_final'].dropna().index.to_numpy()
    angle1 = file1['th_final'].dropna().to_numpy()
    distance1 = file1['d_final'].dropna().to_numpy()
    Pw1 = file1['Pw_Final'].dropna().to_numpy()
    Pv1 = file1['Pv_Final'].dropna().to_numpy()
    
    T_State2 = file2['th_final'].dropna().index.to_numpy()
    angle2 = file2['th_final'].dropna().to_numpy()
    distance2 = file2['d_final'].dropna().to_numpy()
    Pw2 = file2['Pw_Final'].dropna().to_numpy()
    Pv2 = file2['Pv_Final'].dropna().to_numpy()
    
    T_State = np.concatenate((T_State1,T_State2+T_Full1[-1]))
    angle = np.concatenate((angle1,angle2))
    distance = np.concatenate((distance1,distance2))
    Pw = np.concatenate((Pw1,Pw2))
    Pv = np.concatenate((Pv1,Pv2))
    
    
    #pd.merge(file['terminal loss'].dropna(),file['loss'].dropna(),'inner')
    
    #ev = np.isin(T_terminal_loss,T_loss)
    #terminal_loss_small = terminal_loss[ev]
    
    
    ## plot data: loss
    
    plt.style.use("./mystyle2.txt")
    plt.rcParams.update({"text.usetex": True})
    
    ylabel = 'Value'
    xlabel = 'Training epoch'
    title = 'Training loss vs. training epoch'
    fig = plt.figure()
    ax = plt.subplot()
    plt.semilogy(T_loss,loss, c = '#7b3294')
    ax.fill_between(T_loss,loss - integral_loss,(loss - integral_loss)*0,facecolor='#008837')
    ax.fill_between(T_loss,loss,loss - integral_loss,facecolor='#c2a5cf')
    #plt.plot(T_integral_loss,integral_loss)
    #plt.plot(T_terminal_loss,terminal_loss)
    #plt.plot(T_loss, loss - integral_loss)
    
    
    #ax.set_ylim((-4.5,-3))
    #ax.set_xlim((-1,1))
    plt.legend(('Cost $C$', 'Terminal cost $F$','Running cost $r$'))
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+store+'/Loss_'+name1+'.svg')
    
    
    ## plot data: final state
    
    ## plot state data: angle
    plt.style.use("./mystyle2.txt")
    plt.rcParams.update({"text.usetex": True})
    
    ylabel = 'Angle in rad'
    xlabel = 'Training epoch'
    title = 'Average final angle to goal pose'
    fig = plt.figure()
    ax = plt.subplot()
    plt.semilogy(T_State,angle)
    
    #ax.set_ylim((-4.5,-3))
    #ax.set_xlim((-1,1))
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+store+'/Angle_'+name1+'.svg')
    
    
    ## plot state data: distance
    plt.style.use("./mystyle2.txt")
    plt.rcParams.update({"text.usetex": True})
    
    ylabel = 'Distance in m'
    xlabel = 'Training epoch'
    title = 'Average final distance to goal pose'
    fig = plt.figure()
    ax = plt.subplot()
    plt.plot(T_State,distance)
    
    #ax.set_ylim((-4.5,-3))
    #ax.set_xlim((-1,1))
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+store+'/Distance_'+name1+'.svg')
    
    
    ## plot state data: angular momentum
    plt.style.use("./mystyle2.txt")
    plt.rcParams.update({"text.usetex": True})
    
    ylabel = 'Angular momentum in $kg m^2 rad/s$ '
    xlabel = 'Training epoch'
    title = 'Average final angular momentum'
    fig = plt.figure()
    ax = plt.subplot()
    plt.plot(T_State,Pw)
    
    #ax.set_ylim((-4.5,-3))
    #ax.set_xlim((-1,1))
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+store+'/Pw_'+name1+'.svg')
    
    
    
    ## plot state data
    plt.style.use("./mystyle2.txt")
    plt.rcParams.update({"text.usetex": True})
    
    ylabel = 'Linear momentum in $kg m/s$'
    xlabel = 'Training epoch'
    title = 'Average final linear momentum to goal pose'
    fig = plt.figure()
    ax = plt.subplot()
    plt.plot(T_State,Pv)
    
    #ax.set_ylim((-4.5,-3))
    #ax.set_xlim((-1,1))
    ax.set_xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/Figures/'+store+'/Pv_'+name1+'.svg')




##
plt.show()

