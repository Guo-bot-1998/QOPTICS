#%%

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import laguerre
from scipy.special import genlaguerre
import h5py

#%%

def myLG(X,Y,Z,_W0,_Lambda,_L,_P,_path):
    """    myLG.py faster than myLG.m

    myLG.mat |68.680 sec| |121*121*121 points|
    myLG.py |0.564 sec| |121*121*121 points|
    factor: 121.77
    
    MYLG(_W0,_Lambda,Gridz,Gridxy,_L,_P):
    _W0 : Beam Radius at z=0.
    _Lambda: Wavelength of LG beam.
    X, Y, Z: 3-D np.array. x-axis, y-axis cooridnate.
    _L: the azumithal mode number.
    _P: the radial mode number."""

    Zrl = np.pi*_W0**2/_Lambda                         #Rayleigh length
    W= _W0*np.sqrt(1+(Z/Zrl)**2)  
    # Rz = Z + np.divide(Zrl**2, z, out=np.zeros_like(z), where=z!=0.0) #use numpy ufunc
    Rz = Z + Zrl**2/Z 
    Guoy = (abs(_L)+2*_P+1)*np.arctan2(Z,Zrl) 
    
    Nx = X.shape[1]
    Ny = Y.shape[0]
    Nz = Z.shape[2]
    
    LGdata = np.zeros((Nx,Ny,Nz), dtype=np.cfloat)
        
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y,X)
    AL =((np.sqrt(2)*R/W))**abs(_L)
    ALpoly =genlaguerre(_P,abs(_L))(2*(R/W)**2)
    AGauss = np.exp(-(R/W)**2)
    Ptrans1 = np.exp(-1j*(2*np.pi/_Lambda)*R**2/(2*Rz)) # Here
    Ptrans2 = np.exp(-1j*_L*Phi)
    PGuoy = np.exp(1j*Guoy)
    LGdata = (_W0/W)*AL*ALpoly*AGauss*Ptrans1*Ptrans2*PGuoy

    if (_L == 0 and _P == 0):
        Plong = np.exp(-1j*((2*np.pi/_Lambda)*Z - Guoy))
        LGdata = (_W0/W)*AGauss*Ptrans1*Ptrans2*Plong
    
    LGdata = 1*LGdata/np.max(np.abs(LGdata)) 
    
    
    with h5py.File(_path, "w") as f:
        f['LGdata'] = LGdata
        f['MetaParameters/X'] = X
        f['MetaParameters/Y'] = Y
        f['MetaParameters/Z'] = Z
        f['Parameters/W0'] = _W0
        f['Parameters/Lambda'] = _Lambda
        print(f"storing light as {_path}")
        
        
    return LGdata
#%%
def main():
    import numpy as np
    from matplotlib import pyplot as plt
    from numpy.polynomial import laguerre
    from scipy.special import genlaguerre
    import h5py
    import os
    import numpy as np

    Nx = 121
    Ny = 121
    Nz = 121
    Lx = 10
    Ly = 10
    Lz = 10
    x = np.linspace(-Lx,Lx,Nx)
    y = np.linspace(-Ly,Ly,Ny)
    z = np.linspace(-Lz,Lz,Nz)
    L = 10
    P = 0
    W0 = 3.5
    Lambda = 1
    

    [X,Y,Z] = np.meshgrid(x, y, z)
    output = myLG(X,Y,Z, _W0=W0,_Lambda=Lambda,_L=L,_P=P)


#%%
if __name__ == "__main__":
    main()    
