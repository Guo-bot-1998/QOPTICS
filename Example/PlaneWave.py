# #%% example (time stored in 3-th dimension)
# import numpy as np
# from matplotlib import pyplot as plt, animation
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

# fig, ax = plt.subplots()
# x = np.linspace(-3, 3, 91)
# t = np.linspace(0, 25, 30)
# y = np.linspace(-3, 3, 91)
# X3, Y3, T3 = np.meshgrid(x, y, t)
# sinT3 = np.sin(2 * np.pi * T3 / T3.max(axis=2)[..., np.newaxis])
# G = (X3 ** 2 + Y3 ** 2) * sinT3
# cax = ax.pcolormesh(x, y, G[:-1, :-1, 0], vmin=-1, vmax=1)
# fig.colorbar(cax)

# def animate(i):
#    cax.set_array(G[:-1, :-1, i].flatten())

# anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
# anim.save('517.gif')
# plt.show()

#%% use loop to iterate time (3-D)
import numpy as np
from matplotlib import pyplot as plt, animation
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

theta = np.pi/2    # spherical coordinates 
phi = np.pi/4

unit = 1e-10
wl = 2*np.pi * 1e-10
wl = wl/unit
k = 2*np.pi/wl

#kx, ky, kz
K = (k*np.sin(theta)*np.cos(phi), k*np.sin(theta)*np.sin(phi), k*np.cos(theta))
Amp = 10
x = np.linspace(-2*np.pi, 2*np.pi,  300)
y = np.linspace(-2*np.pi, 2*np.pi, 300)
z = np.linspace(-4*np.pi, 4*np.pi, 50)
X,Y,Z = np.meshgrid(x,y,z)

T = 2
w = 2*np.pi/T
total = T
dt = 0.1
t = np.linspace(0, total, int(total/dt)+1)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
for i,ti in enumerate(t):
    wave = Amp*np.exp(1j*(K[0]*X+K[1]*Y+K[2]*Z - w*ti))
    # print(np.max(wave[:,:,0]))
    # plt.pcolormesh(x,y,np.real(wave[:, :, 0]))
    ax.pcolormesh(x,y, np.real(wave[:, :, 0])) 
    
    plt.pause(0.0001)
    ax.clear()
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    
    


# %%
