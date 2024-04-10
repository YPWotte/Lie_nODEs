#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:34:15 2023

@author: yannik
Source code adapted from:
    Nicolas P. Rougier. Scientific Visualization: Python + Matplotlib. Nicolas P. Rougier. 2021, 978-2-
9579901-0-8. ffhal-03427242
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

### Function to retrieve data X,Y
def simulate():
    d = 0.005
    # x = np.random.uniform(0, d)
    # y = d - x
    x, y = np.random.uniform(0, d, 2)

    dt = 0.05
    t = 35.0
    alpha = 0.25
    n = int(t / dt)
    X = np.zeros(n)
    Y = np.zeros(n)
    C = np.random.randint(0, 2, n)

    for i in range(n):
        # Asynchronous
        if 0:
            if C[i]:
                x += (alpha + (x - y)) * (1 - x) * dt
                x = max(x, 0.0)
                y += (alpha + (y - x)) * (1 - y) * dt
                y = max(y, 0.0)
            else:
                y += (alpha + (y - x)) * (1 - y) * dt
                y = max(y, 0.0)
                x += (alpha + (x - y)) * (1 - x) * dt
                x = max(x, 0.0)
        # Synchronous
        else:
            dx = (alpha + (x - y)) * (1 - x) * dt
            dy = (alpha + (y - x)) * (1 - y) * dt
            x = max(x + dx, 0.0)
            y = max(y + dy, 0.0)
        X[i] = x
        Y[i] = y
    return X, Y


### Initial simulations, S[i] stores i-th batch of type X,Y
np.random.seed(11)
S = []
n = 250
for i in range(n):
    S.append(simulate())



### Plot 1: many lines
plt.figure(figsize=(20, 10))
ax = plt.subplot(121, aspect=1) # Left subplot
axins = zoomed_inset_axes(ax, 25, loc=3) # Zoomed-in insertion in left subplot

## Loop over S, plot various batches of type X,Y individually
for i in range(n):
    X, Y = S[i]
    if X[-1] > 0.9 and Y[-1] > 0.9: # Color one type of trajectories AND insert dots for their starting positions
        c = "r"
        lw = 1.0
        axins.scatter(X[0], Y[0], c="r", edgecolor="w", zorder=10)
    else:                           # Color the other type of trajectories
        c = "b"
        lw = 1.0
    ax.plot(X, Y, c=c, alpha=0.25, lw=lw) # Plot all into both ax and axins
    axins.plot(X, Y, c=c, alpha=0.25, lw=lw) 

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_title("%d trajectories of a dual particle system (x,y)" % n)
axins.set_xlim(0.01, 0.02) # limit the small plot's x and y axis to make sure it is not strange
axins.set_xticks([])
axins.set_ylim(0.01, 0.02) 
axins.set_yticks([])

ax = plt.subplot(122, aspect=1)
axins = zoomed_inset_axes(ax, 50, loc=3)
axins.set_facecolor((1, 1, 0.9))
n = 9
for i in range(n):
    X, Y = S[i]
    ls = "-"
    if i == 2:
        ls = "--"
    if X[-1] > 0.9 and Y[-1] > 0.9:
        c = "r"
        lw = 2.0
        axins.scatter(X[0], Y[0], s=150, c="r", edgecolor="w", zorder=10, lw=2)
    else:
        c = "b"
        lw = 2.0
    ax.plot(X, Y, c=c, alpha=0.75, lw=lw, ls=ls)
    axins.plot(X, Y, c=c, alpha=0.75, lw=lw, ls=ls)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["0", "1"], fontsize=16)
ax.set_yticklabels(["0", "1"], fontsize=16)
ax.set_xlabel("x position", fontsize=20)
ax.set_ylabel("y position", fontsize=20)
# ax.set_title('%d trajectories of a dual particle system (x,y)' % n)
axins.set_xlim(0.01, 0.02)
axins.set_xticks([])
axins.set_ylim(0.01, 0.02)
axins.set_yticks([])

plt.savefig("./Results//Figures/Tests/rule-3.pdf")
plt.show()