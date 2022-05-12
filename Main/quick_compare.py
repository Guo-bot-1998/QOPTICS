# quick plotting
#%%
import numpy as np
import utils.dirutils as dd,os
import sys
from utils.dummyconst import *

# def Hamiltonian(_psi, _G, _dx, _dy, _dz):
#      Energy = (np.sum( (np.conjugate(_psi) *  
#          (-0.5 *del2.del2(_psi,_dx,_dy,_dz)+(Epot + _G*np.abs(_psi)**2)*_psi)*_dx*_dy*_dz)))

#      return Energy
#%%
data = dd.retrieve()
module = sys.modules[__name__]
for name, value in data.items():
    setattr(module, name, value)
if 'LGdata' in globals(): # The light is stored as variable named 'LGdata'
    LG = LGdata
#%%

x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
z = np.linspace(-Lz,Lz,Nz)
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
X,Y,Z = np.meshgrid(x,y,z)
# Some constants
pi = 3.14159265359
hbar = 1.054571800139113e-34 
m = 1.411000000000000e-25  # Rb atoms
# BEC parameters
As = 5.82e-09
Nbec = 10000
Rabi = 1000
Wx = 2000
Wy = 2000
Wz = 2000
# unit = 1.222614572474304e-06;
unit = np.sqrt(hbar/m/Wz)

Ggg = (4*pi*hbar**2*As*Nbec/m)*unit**-3*(hbar*Wz)**-1
Gee = Ggg  
Gge = 0
Geg = 0
Epot = ( (Wx**2*X**2 + Wy**2*Y**2 + Wz**2*Z**2 )
            / (2*Wz**2) )
# circular potential
psiGmu = (15*Ggg / ( 16*pi*np.sqrt(2) )  )**(2/5)  
psiEmu = (15*Gee / ( 16*pi*np.sqrt(2) )  )**(2/5) 



# %%
TF_amp = np.array((psiGmu-Epot)/Ggg,dtype=np.cfloat)
np.clip(TF_amp, 0, np.inf,out=TF_amp)
TF_pbb = np.sqrt(TF_amp)
total = np.sum(np.abs(TF_pbb)**2*dx*dy*dz)
n_TF_pbb = TF_pbb/np.sqrt(total)

#%%
print(np.sum(np.abs(TF_pbb)**2*dx*dy*dz))
print(np.sum(np.abs(psiG)**2*dx*dy*dz))
print(np.sum(np.abs(psiE)**2*dx*dy*dz))

#%%
print("abs|n_TF_pbb|^2")
print(np.sum(np.abs(n_TF_pbb)**2*dx*dy*dz))
print("abs|psiG|^2")
print(np.sum(np.abs(psiG)**2*dx*dy*dz))
print("abs|psiE|^2")
print(np.sum(np.abs(psiE)**2*dx*dy*dz))
#%%
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x, np.abs(n_TF_pbb[61,:,61])**2*dx*dy*dz)
plt.plot(x, np.abs(psiG[61,:,61])**2*dx*dy*dz)
plt.xlabel("x")
plt.title("psiG")
plt.legend(["Thomas-Fermi", "psiG"])

plt.figure()
plt.plot(y, np.abs(n_TF_pbb[:,61,61]**2)*dx*dy*dz)
plt.plot(y, np.abs(psiG[:,61,61]**2)*dx*dy*dz)
plt.xlabel("y")

plt.figure()
plt.plot(z, np.abs(n_TF_pbb[61,61,:]**2)*dx*dy*dz)
plt.plot(z, np.abs(psiG[61,61,:]**2)*dx*dy*dz)
plt.xlabel("z")

plt.figure()
plt.plot(x, np.abs(n_TF_pbb[61,:,61])**2*dx*dy*dz)
plt.plot(x, np.abs(psiE[61,:,61])**2*dx*dy*dz)
plt.xlabel("x")
plt.title("psiE")
plt.legend(["Thomas-Fermi", "psiG"])

plt.figure()
plt.plot(y, np.abs(n_TF_pbb[:,61,61]**2)*dx*dy*dz)
plt.plot(y, np.abs(psiE[:,61,61]**2)*dx*dy*dz)
plt.xlabel("y")

plt.figure()
plt.plot(z, np.abs(n_TF_pbb[61,61,:]**2)*dx*dy*dz)
plt.plot(z, np.abs(psiE[61,61,:]**2)*dx*dy*dz)
plt.xlabel("z")

plt.figure()
plt.plot(x, np.abs(LG[61,:,61])**2*dx*dy*dz)
plt.xlabel("x")
plt.title("LG")
#


# %%
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
cut = 60
cut2 = 41
phase = np.arctan2(np.imag(psiE[...]),np.real(psiE[...]))
fig, axs =plt.subplots(1,2)
cont = axs[0].pcolor(X[..., cut],Y[..., cut], phase[...,cut])
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].axhline(y=y[cut2],color='r',linestyle='-')
plt.colorbar(cont, ax=axs[0])
axs[0].set_title("phase of psiE at z=0")

axs[1].plot(x,phase[cut2,:,cut])
axs[1].set_xlabel('x')
axs[1].set_ylabel('phase')
axs[1].set_title("phase of psiE at redline")

# %%
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
cut = 60
cut2 = 51
density = np.abs(psiG)**2*dx*dy*dz
fig, axs =plt.subplots(1,2)
cont = axs[0].pcolor(X[..., cut],Y[..., cut], density[...,cut])
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].axhline(y=y[cut2],color='r',linestyle='-')
plt.colorbar(cont, ax=axs[0])
axs[0].set_title("density of psiG at z=0")

axs[1].plot(x,density[cut2,:,cut])
axs[1].set_xlabel('x')
axs[1].set_ylabel('density')
axs[1].set_title(f"density of psiG at y={y[cut2]}")
# %% intensity
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(x=x,y=y,z=np.abs(psiG[:,:,cut]**2*dx*dy*dz))])
fig.update_layout(title='psiG_intensity', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
fig = go.Figure(data=[go.Surface(x=x,y=y,z=np.abs(psiE[:,:,cut]**2*dx*dy*dz))])
fig.update_layout(title='psiE_intensity', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

fig = go.Figure(data=[go.Surface(x=x,y=y,z=np.abs(LG[:,:,cut]**2*dx*dy*dz))])
fig.update_layout(title='LG_intensity', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
# %% phase
import plotly.graph_objects as go
phase = np.arctan2(np.imag(psiG[:,:,cut])
                  ,np.real(psiG[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=x,y=y,z=phase)])
fig.update_layout(title='psiG_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

phase = np.arctan2(np.imag(psiE[:,:,cut])
                  ,np.real(psiE[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=x,y=y,z=phase)])
fig.update_layout(title='psiE_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

phase = np.arctan2(np.imag(LG[:,:,cut])
                  ,np.real(LG[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=x,y=y,z=phase)])
fig.update_layout(title='LG_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()


#%% LG cross-line along xy-plane
from matplotlib import pyplot as plt

dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
X,Y,Z = np.meshgrid(x,y,z)

cut = 60
plt.figure()
plt.plot(x, np.abs(LGdata[60,:,cut])**2*dx*dy*dz)
plt.pcolor(np.abs(LGdata[:,:,cut]))

#%% LG intensity from python
cut = 61
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=np.abs(LGdata[:,:,cut]))])
fig.update_layout(title="amp", autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

#%% LG phase from python
cut = 0
import plotly.graph_objects as go
phase = np.arctan2(np.imag(LGdata[:,:,cut])
                  ,np.real(LGdata[:,:,cut]))
fig = go.Figure(data=[go.Surface(z=phase)])
fig.update_layout(title=' phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()


#%%
# print('\nisosurface\n')
# import plotly.graph_objects as go
# Data = np.abs(psiE**2*dx*dy*dz).flatten()
# # [X,Y,Z] = np.meshgrid(x,y,z)
# # Data = (X**2+Y**2+Z**2).flatten()
# diff = np.max(Data) - np.min(Data)
# fig= go.Figure(data=go.Isosurface(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value = Data,
#     # isomin=np.abs(Data[61,61,61]),
#     # isomax=np.abs(Data[50,50,61]),
#     isomin=np.min(Data) + diff/5,
#     isomax=np.max(Data) - diff/5,
#     # isomin=1e-5,
#     # isomax=3e-5,
# ))
# fig.show()

# import plotly.graph_objects as go

# fig= go.Figure(data=go.Isosurface(
#     x=[0,0,0,0,1,1,1,1],
#     y=[1,0,1,0,1,0,1,0],
#     z=[1,1,0,0,1,1,0,0],
#     value=[1,2,3,4,5,6,7,8],
#     isomin=2,
#     isomax=6,
# ))