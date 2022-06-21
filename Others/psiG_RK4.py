# RK4 to iterate time in Hamiltonian. Not test yet
# (may not be used)
from numba import njit
# %%
@njit(fastmath=True, nogil=True)
def compute_BEC_RK4(_psiG, _psiE, nj):
    """Calculating interaction between the BEC and L.G. beams.
    Two-order system is used. The code evaluates ground-
    state BEC and excited-state BEC, and save the data.
    Note: Data is calculated without units. Use RK4 method to update
    time.
   
    Args:
        nj: Number of iterations.
        stepJ: Number of iterations to update energy constraint.
        isLight: interaction with light?.
        x,y,z: coordinate vectors
        dw: finite time difference.
    """
    
    # Potential and initial condition
    _psiGmu = psiGmu
    # _psiEmu = psiEmu
         - (Epot + Ggg*np.abs(_psiG)**2 + Gge*np.abs(_psiE)**2)*_psiG
         + _psiGmu*_psiG )
    # set boundary points to zero
    tmp[0,:,:] = tmp[-1,:,:] = tmp[:,-1,:] = tmp[:,0,:] =  tmp[:,:,0] = tmp[:,:,-1] = 0
    return tmp
    
@njit(fastmath=True, nogil=True)
def E(_psiG, _psiE, _psiEmu):
    """Function of excited state"""
    tmp = np.zeros(_psiE.shape)
    tmp = ( 0.5 * del2(_psiE,dx,dy,dz) 
         - (Epot + Gee*np.abs(_psiE)**2 + Geg*np.abs(_psiG)**2)*_psiE
         + _psiEmu*_psiE )
    # set boundary points to zero
    tmp[0,:,:] =  tmp[-1,:,:] = tmp[:,-1,:] = tmp[:,0,:] =  tmp[:,:,0] = tmp[:,:,-1] = 0
    return tmp

@njit(fastmath=True, nogil=True)
def RK4(_psiG, _psiE, _psiGmu, _psiEmu, h):
    k_g = np.zeros((*_psiG.shape,4))
    k_e = np.zeros((*_psiG.shape,4))
    k_g[...,0] = G(_psiG, _psiE, _psiGmu) 
    k_e[...,0] = E(_psiG, _psiE, _psiEmu) 
    k_g[...,1] = G(_psiG + h*k_g[...,1]/2, _psiE + h*k_e[...,1]/2, _psiGmu) 
    k_e[...,1] = E(_psiG + h*k_g[...,1]/2, _psiE + h*k_e[...,1]/2, _psiEmu) 
    k_g[...,2] = G(_psiG + h*k_g[...,2]/2, _psiE + h*k_e[...,2]/2, _psiGmu) 
    k_e[...,2] = E(_psiG + h*k_g[...,2]/2, _psiE + h*k_e[...,2]/2, _psiEmu) 
    k_g[...,3] = G(_psiG + h*k_g[...,3], _psiE + h*k_e[...,3], _psiGmu) 
    k_e[...,3] = E(_psiG + h*k_g[...,3], _psiE + h*k_e[...,3], _psiEmu) 
    _psiG = _psiG + h/6 * ( k_g[...,0] + 2*k_g[...,1] + 2*k_g[...,2] + k_g[...,3] )
    _psiE = _psiE + h/6 * ( k_e[...,0] + 2*k_e[...,1] + 2*k_e[...,2] + k_e[...,3] )

    return _psiG, _psiE
