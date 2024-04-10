#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:05:05 2023

@author: yannik
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("nbAgg")

fig = plt.figure(figsize=(7,2), dpi=100)
ax = plt.subplot()
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
line1, = ax.plot(X, C, marker="o", markevery=[-1],
markeredgecolor="white")
line2, = ax.plot(X, S, marker="o", markevery=[-1],
markeredgecolor="white")
def update(frame):
    line1.set_data(X[:frame], C[:frame])
    line2.set_data(X[:frame], S[:frame])
#ani = animation.FuncAnimation(fig, update, interval=10)
# plt.show()
# plt.savefig('./Results/Figures/Tests/test.png')

writer = animation.FFMpegWriter(fps=30)
anim = animation.FuncAnimation(fig, update, interval=10, frames= len(X))
# anim.save("./Results/Figures/Tests/sine-cosine.mp4", writer=writer, dpi=100)

from tqdm.autonotebook import tqdm
bar = tqdm(total=len(X))
anim.save("./Results/Figures/Tests/sine-cosine-2.mp4", writer=writer, dpi=300,
progress_callback = lambda i, n: bar.update(1))
bar.close()