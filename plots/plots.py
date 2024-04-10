#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:26:40 2023

@author: yannik
"""

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## lietorch:
import sys; sys.path.append('../')
import lie_torch as lie
import math

## Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mpl.rcParams.update(mpl.rcParamsDefault)

class Plot:
    def __init__(self, model,prior):
        self.model = model
        self.prior = prior
        self.f = self.model.f
        
    def compute(self,T):
        self.T = T
        batch_size = 100;
        t_span = torch.linspace(0, T, 1000).to(device);

        x0 = self.prior.sample((batch_size)).to(device)
        t, xT = self.model.forward(x0,t_span)
        self.t = t
        self.xT = xT
        
    def angle(self,T):
        if self.T < T:
            self.compute(T)
        t = self.T
        xT = self.xT
        
        plt.style.use("./mystyle.txt")
        plt.rcParams.update({"text.usetex": True})
        fig1 = plt.figure()#figsize=(10, 10))
        ax1 = plt.subplot()#aspect=1) 

        for i in range(100):
            xiT = xT[:,i,:]
            T = t.detach().numpy()
            th, d = self.target.distance(xiT)
            th = th.detach().numpy()
            d = d.detach().numpy()
            c = "r"
            lw = 1.0
            ax1.plot(T, th, c=c, alpha=0.25, lw=lw) 

        ax1.set_xlim(0, T)
        ax1.set_ylim(0, np.pi)
        ax1.set_xlabel("Time in s")
        ax1.set_ylabel("Angle in rad")
        ax1.set_title("Angle to goal pose for %d trajectories of rigid bodies" % 100)
        ax1.set_yticks([0, np.pi/2, np.pi])
        ax1.set_yticklabels(["0", "$\pi$/2", "$\pi$"])
        #ax1.set_xticks([0, 1, 2, 3, 4 , 5])
        #ax1.set_xticklabels(["0", "1", "2", "3", "4", "5"], usetex=True)
        plt.savefig('./Results/Figures/Angle_v_Time_%ds.png' %T)
        
    def dist(self,T):
        if self.T < T:
            self.compute(T)
        t = self.T
        xT = self.xT
        
        plt.style.use("./mystyle.txt")
        plt.rcParams.update({"text.usetex": True})
        fig2 = plt.figure()#figsize=(10, 10))
        ax2 = plt.subplot()#aspect=1) 

        for i in range(100):
            xiT = xT[:,i,:]
            T = t.detach().numpy()
            th, d = self.target.distance(xiT)
            th = th.detach().numpy()
            d = d.detach().numpy()
            c = "r"
            lw = 1.0
            ax2.plot(T, d, c=c, alpha=0.25, lw=lw)

        ax2.set_xlim(0, T)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time in s")
        ax2.set_ylabel("Distance in m")
        ax2.set_title("Distance to goal pose for %d trajectories of rigid bodies" % 100)
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        #ax2.set_xticks([0, 1, 2, 3, 4 , 5])
        #ax2.set_xticklabels(["0", "1", "2", "3", "4", "5"], usetex=True)
        plt.savefig('./Results/Figures/Distance_v_Time_%ds.png' %T)
    
    def allplots(self,T):
        
        if self.T < T:
            self.compute(T)
        t = self.T
        xT = self.xT
        
        plt.style.use("./mystyle.txt")
        plt.rcParams.update({"text.usetex": True})
        fig1 = plt.figure()#figsize=(10, 10))
        ax1 = plt.subplot()#aspect=1) 

        fig2 = plt.figure()#figsize=(10, 10))
        ax2 = plt.subplot()#aspect=1) 

        for i in range(100):
            xiT = xT[:,i,:]
            T = t.detach().numpy()
            th, d = self.target.distance(xiT)
            th = th.detach().numpy()
            d = d.detach().numpy()
            c = "r"
            lw = 1.0
            ax1.plot(T, th, c=c, alpha=0.25, lw=lw) 
            ax2.plot(T, d, c=c, alpha=0.25, lw=lw)

        ax1.set_xlim(0, T)
        ax1.set_ylim(0, np.pi)
        ax1.set_xlabel("Time in s")
        ax1.set_ylabel("Angle in rad")
        ax1.set_title("Angle to goal pose for %d trajectories of rigid bodies" % 100)
        ax1.set_yticks([0, np.pi/2, np.pi])
        ax1.set_yticklabels(["0", "$\pi$/2", "$\pi$"])
        #ax1.set_xticks([0, 1, 2, 3, 4 , 5])
        #ax1.set_xticklabels(["0", "1", "2", "3", "4", "5"], usetex=True)
        plt.savefig('./Results/Figures/Angle_v_Time_%ds.png' %T)

        ax2.set_xlim(0, T)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time in s")
        ax2.set_ylabel("Distance in m")
        ax2.set_title("Distance to goal pose for %d trajectories of rigid bodies" % 100)
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        #ax2.set_xticks([0, 1, 2, 3, 4 , 5])
        #ax2.set_xticklabels(["0", "1", "2", "3", "4", "5"], usetex=True)
        plt.savefig('./Results/Figures/Distance_v_Time_%ds.png' %T)
        
        
    def LinePotential(self):
        plt.style.use("./mystyle.txt")
        plt.rcParams.update({"text.usetex": True})

        N = 1000
        w = torch.rand(3)
        v = torch.rand(3);

        w = w/lie.angle(w);
        v = v/lie.angle(v);
        t = torch.cat((w,v))
        V = np.zeros((1000));
        th = np.zeros((1000));
        wn = torch.rand(1)

        ## Distance only, arbitrary angle
        for i in range(N):
            #ti = t*torch.pi/N*i;
            #ti = t*i/N;
            ti = torch.cat((w*wn,v*i/N))
            H = lie.exp_SE3(ti);
            V[i] = self.f.Potential(H)
            #th[i] = lie.angle(ti)
            th[i] = lie.angle(ti[3:])

        ylabel = 'Potential'
        xlabel = 'Distance'
        title = 'Potential vs. Distance, arbitrary angle'
        fig = plt.figure()
        plt.plot(th,V)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig('./Results/Figures/Pot_v_Dist.png')

        ## Distance and angle
        for i in range(N):
            #ti = t*torch.pi/N*i;
            ti =  torch.cat((w*torch.pi,v))*i/N;
            #ti = torch.cat((w*wn,v*i/N))
            H = lie.exp_SE3(ti);
            V[i] = self.f.Potential(H)
            #th[i] = lie.angle(ti)
            th[i] = lie.angle(ti[3:])

        ylabel = 'Potential'
        xlabel = 'Distance'
        title = 'Potential vs. Screw'
        plt.figure()
        plt.plot(th,V)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig('./Results/Figures/Pot_v_Screw.png')

        ## Angle only, zero distance
        for i in range(N):
            ti = torch.cat((w*2*torch.pi*i/N,v*0))
            H = lie.exp_SE3(ti);
            V[i] = self.f.Potential(H)
            th[i] = lie.angle(ti)

        ylabel = 'Potential'
        xlabel = 'Angle'
        title = 'Potential vs. Angle'
        plt.figure()
        plt.plot(th,V)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig('./Results/Figures/Pot_v_Ang.png')


        ## Distance outside training range:

        for i in range(N):
            #ti = t*torch.pi/N*i;
            #ti = t*i/N;
            ti = torch.cat((w*0,v*50*i/N))
            H = lie.exp_SE3(ti);
            V[i] = self.f.Potential(H)
            #th[i] = lie.angle(ti)
            th[i] = lie.angle(ti[3:])

        ylabel = 'Potential'
        xlabel = 'Distance'
        title = 'Potential vs. Distance outside training range'
        plt.figure()
        plt.plot(th,V)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig('./Results/Figures/Pot_v_Dist_outside_range.png')
