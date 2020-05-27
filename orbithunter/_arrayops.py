import numpy as np

__all__ = ['swap_modes', 'so2_generator', 'so2_coefficients']

def swap_modes(modes, dimension='space'):
    if dimension == 'space':
        m = modes.shape[1]//2
        swapped_modes = np.concatenate((modes[:,-m:],modes[:,:-m]),axis=1)
    else:
        n = (modes.shape[0]+1)//2 - 1
        swapped_modes = np.concatenate((modes[0,:].reshape(1,-1), modes[-n:,:], modes[1:-n,:]),axis=0)
    return swapped_modes

def so2_generator(order=1):
    return np.linalg.matrix_power(np.array([[0,-1],[1,0]]), np.mod(order,4))

def so2_coefficients(order=1):
    return np.sum(so2_generator(order=order),axis=0)


