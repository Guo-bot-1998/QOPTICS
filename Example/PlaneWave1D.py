# %% 1 Dimension

import numpy as np
from matplotlib import pyplot as plt, animation
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


unit = 1e-10

wl = 4 * 1e-10
wl = wl/unit
k = 2*np.pi/wl
Amp = 10
x = np.linspace(-10*np.pi, 10*np.pi,  1000)

T = 2
w = 2*np.pi/T
total = 5*T
dt = 0.1
t = np.linspace(0, total, int(total/dt)+1)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(x,x) 
for i,ti in enumerate(t):
    wave = Amp*np.exp(1j*(k*x - w*ti))
    line.set_ydata(wave)
    fig.canvas.draw()
    plt.pause(1)
    fig.canvas.flush_events()
