import logging
import jax
import jax.numpy as jnp
import numpy as np
import functools
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import utils

# Some functions that do linear algera on batch in GPU. I find it strange that there is not already a decent library to do this, but I couldn't find one.

logger = logging.getLogger(__name__)

def batch_st_end(k, batch_size, n_rows):
    batch_st = int(k * batch_size)
    batch_end = int(np.min( [(k+1) * batch_size, n_rows]))
    return batch_st, batch_end

def blockwise_Y_T_X(Y,X, batch_size = None, memory_to_use = 10):
    # X and Y are tall and skinny
    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        size_of_Y = utils.get_size_in_gb(Y)
        n_blocks = np.ceil(4 * (size_of_X + size_of_Y) / memory_to_use).astype(int)
        batch_size = np.floor(X.shape[0] / n_blocks)
        
    n_rows = X.shape[0]
    YX = jnp.zeros_like(X, shape =[Y.shape[-1], X.shape[-1]])
    square_jit = jax.jit( lambda y, x: jnp.conj(y).T @ x)
    logging.info(f"Y^T @ X {int(np.ceil(n_rows/batch_size))} blocks") 
    for k in range(0, int(np.ceil(n_rows/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        YX += square_jit(Y[batch_st:batch_end], X[batch_st:batch_end]) #jnp.conj(Z).T @ Z
        # utils.report_memory_device(logger =logger)
    return np.array(YX)


def blockwise_X_T_X(X, batch_size = None, memory_to_use = 10):
    # X
    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        n_blocks = np.ceil(size_of_X / memory_to_use).astype(int)
        batch_size = np.floor(X.shape[0] / n_blocks)
        
    n_rows = X.shape[0]
    XX = jnp.zeros_like(X, shape =[X.shape[-1], X.shape[-1]])
    square_jit = jax.jit( lambda x: jnp.conj(x).T @ x)
    
    for k in range(0, int(np.ceil(n_rows/batch_size))):
        print(str(k)+",", end="") 
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        Z = jnp.array(X[batch_st:batch_end])
        XX += square_jit(Z) #jnp.conj(Z).T @ Z
        utils.report_memory_device(logger =logger)

    return np.array(XX)        
            


def blockwise_A_X(A, X, batch_size = None, memory_to_use = 10):
    # Blockwise multiply where # A is very tall, and X is square-ish

    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        usable_memory = memory_to_use - size_of_X 
        # max_item_size = A.itemsize if A.itemsize > X.itemsize else X.itemsize
        max_item_size = (A[0,0] * X[0,0]).itemsize # item size of product

        size_of_A = A.shape[0]* ( np.max([A.shape[1], X.shape[1]])) * max_item_size / 1e9
        n_blocks = np.ceil( 4 * size_of_A / usable_memory).astype(int)
        batch_size = np.floor(A.shape[0] / n_blocks)
        
    n_rows = A.shape[0]
    # Compute the bottom of fraction.
    Z = np.zeros_like(A, shape = [A.shape[0], X.shape[-1]])
    utils.report_memory_device(logger =logger)
    X = jnp.array(X)
    
    mat_mat_jit = jax.jit( lambda x, y: x @ y)
    logging.info(f"A@X in {int(np.ceil(n_rows/batch_size))} blocks") 
    for k in range(0, int(np.ceil(n_rows/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        Z[batch_st:batch_end] = np.array(mat_mat_jit( A[batch_st:batch_end],  X))
    return Z        



# def blockwise_X_A(X, A, batch_size = None, memory_to_use = 10):
#     # Blockwise multiply where # A is very tall, and X is square-ish

#     if batch_size is None:
#         size_of_X = utils.get_size_in_gb(X)
#         usable_memory = memory_to_use - size_of_X 
#         size_of_A = utils.get_size_in_gb(A)
#         n_blocks = np.ceil( 4 * size_of_A / usable_memory).astype(int)
#         batch_size = np.floor(A.shape[0] / n_blocks)
        
#     n_rows = A.shape[0]
#     # Compute the bottom of fraction.
#     Z = np.zeros_like(A, shape = [X.shape[0], A.shape[-1]])
#     utils.report_memory_device(logger =logger)
#     X = jnp.array(X)
    
#     mat_mat_jit = jax.jit( lambda x, y: x @ y)
#     logging.info(f"A@X in {int(np.ceil(n_rows/batch_size))} blocks") 
#     for k in range(0, int(np.ceil(n_rows/batch_size))):
#         batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
#         Z = np.array(mat_mat_jit(X,  A[batch_st:batch_end] ))
#     return Z        
            
    
# These two methods are not used in the main code because they are a bit gross, 
# but still used in some old part that are useful for debugging so I'll leave them for now
def thin_svd_in_blocks(X, np = np, memory_to_use = 5, epsilon = 1e-8, n_components = -1):
    '''
    This is an unstable method for SVD but that can run on GPU.
    This should be only (maybe) used for matrices that cannot fit on the GPU.
    
    Same as SVD but dispatch matrices in blocks to gpu (also pretty unstable)
    '''
    
    Y = blockwise_X_T_X(X, memory_to_use = memory_to_use)
    Ys, Yu = np.linalg.eigh(Y)#, full_matrices = True)
    sigma = np.sqrt(np.where(Ys > 0, Ys, 0))
    # Avoid an annoying warning?
    sigma_pos = np.where( sigma > epsilon , sigma, epsilon)
    sigma_inv = np.where( sigma > epsilon , 1/ sigma_pos, 0)
    
    U = blockwise_A_X(X, Yu, memory_to_use = memory_to_use)
    U = U * sigma_inv
    V = Yu
    
    n_components = n_components if n_components > 0 else X.shape[0]
    
    return np.flip(U, axis =1)[:,:n_components], np.flip(sigma)[:n_components], np.conj(np.flip(V, axis =1))[:,:n_components].T



def thin_svd(X, np = np, epsilon = 1e-8):
    '''
    For some reason, the built in svd seems to allocate a lot more memory than necessary.
    '''
    Y = np.conj(X).T @ X
    Ys, Yu = np.linalg.eigh(Y)#, full_matrices = True)
    sigma = np.sqrt(np.where(Ys > 0, Ys, 0))
    sigma_inv = np.where( sigma > epsilon , 1/ sigma, 0)
    U = (X @ Yu) * sigma_inv
    V = Yu
    return np.flip(U, axis =1), np.flip(sigma), np.flip(V, axis =1)


def randomized_svd(A, n_pcs = 200):
    '''
    For some reason, the built in svd seems to allocate a lot more memory than necessary.
    '''
    n_pcs = n_pcs if n_pcs < A.shape[1] else A.shape[1]
    gauss = np.random.randn(A.shape[1], n_pcs)
    Agauss = blockwise_A_X(A, gauss, memory_to_use = utils.get_gpu_memory_total()//3)
    qr_cpu = jax.jit(jnp.linalg.qr, backend='cpu')
    Q, _ = qr_cpu(Agauss)
    logger.info("QR done")
    Y = blockwise_Y_T_X(Q,A) #np.conj(Q).T @ A
    logger.info("Q^TA done")
    svd_cpu = jax.jit(jnp.linalg.svd, backend='cpu')
    U, S, Vh = svd_cpu(Y)#, full_matrices = True)
    QU = blockwise_A_X(Q, U, memory_to_use = utils.get_gpu_memory_total()//3)

    return QU, S, Vh

    # Y = np.conj(X).T @ X
    # Ys, Yu = np.linalg.eigh(Y)#, full_matrices = True)
    # sigma = np.sqrt(np.where(Ys > 0, Ys, 0))
    # sigma_inv = np.where( sigma > epsilon , 1/ sigma, 0)
    # U = (X @ Yu) * sigma_inv
    # V = Yu
    # return np.flip(U, axis =1), np.flip(sigma), np.flip(V, axis =1)



#### batching IDFT

# Assumes input are of size (vol_size, n_vol)
# This seems like a crazy amount of reshaping/transposing
# ### REWRITE THIS TRASH
@functools.partial(jax.jit, static_argnums = [1])    
def idft3(x, vec_shape ):
    x = x.reshape([*vec_shape, x.shape[-1]]) 
    x = x.T
    x = ftu.get_idft3(x)
    x = x.T
    x = x.reshape([-1, x.shape[-1]])
    return x

@functools.partial(jax.jit, static_argnums = [1])    
def dft3(x, vec_shape):
    x = x.reshape([*vec_shape, x.shape[-1]])
    x = x.T
    x = ftu.get_dft3(x)
    x = x.T
    x = x.reshape([-1, x.shape[-1]])
    return x

def batch_idft3(x, vec_shape, batch_size):
    x_out = np.zeros_like(x) 
    n_tot = x.shape[-1]
    logging.info(f"batch_idft3 in {int(np.ceil(n_tot/batch_size))} blocks") 
    for k in range(0, int(np.ceil(n_tot/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_tot)
        x_out[:,batch_st:batch_end] = np.array(idft3(x[:,batch_st:batch_end], vec_shape = vec_shape))
    return x_out
    
    
def batch_dft3(x, vec_shape, batch_size):
    x_out = np.zeros_like(x, dtype = 'complex64') 
    n_tot = x.shape[-1]
    logging.info(f"batch_idft3 in {int(np.ceil(n_tot/batch_size))} blocks") 
    for k in range(0, int(np.ceil(n_tot/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_tot)
        x_out[:,batch_st:batch_end] = np.array(dft3(x[:,batch_st:batch_end], vec_shape = vec_shape))
    return x_out

def broadcast_dot(x,y):
    return jax.lax.batch_matmul(jnp.conj(x[...,None,:]),y[...,:,None])[...,0,0]

def broadcast_outer(x,y):
    return jax.lax.batch_matmul(x[...,:,None],jnp.conj(y[...,None,:]))

def multiply_along_axis(A, B, axis):
    return jnp.swapaxes(jnp.swapaxes(A, axis, -1) * B, -1, axis)


def batch_hermitian_linear_solver(A,b):
    return jax.scipy.linalg.solve(A,b, assume_a = 'pos')

def batch_linear_solver(A,b):
    return jax.scipy.linalg.solve(A,b)

# Maybe if problems come again...
#     if A.ndim ==2:
#         return lu_solve(A,b)
#     return batched_solve_lu(A,b)

# def Cholesky_solve(A,b):
#     # I'll call lineax because it throws a proper error message
#     return lx.linear_solve(lx.MatrixLinearOperator(A), b, solver=lx.Cholesky()).value

# def lu_solve(A,b):
#     # I'll call lineax because it throws a proper error message
#     return lx.linear_solve(lx.MatrixLinearOperator(A), b, solver=lx.LU()).value

# import lineax as lx
# batched_solve_lu = jax.vmap( lambda matrix, vector: lx.linear_solve(lx.MatrixLinearOperator(matrix), vector, solver=lx.LU()).value)
# batched_solve_Cholesky = jax.vmap( lambda matrix, vector: lx.linear_solve(lx.MatrixLinearOperator(matrix), vector, solver=lx.Cholesky()).value)




def solve_by_SVD(A,b, hermitian = False):
    U,S,Vh = jax.numpy.linalg.svd(A, hermitian = False)

    if b.ndim == A.ndim -1:
        expand = True
        b = b[...,None]
    else:
        expand = False
    
    Uhb = jax.lax.batch_matmul(jnp.conj(U.swapaxes(-1,-2)),b)/ S[...,None]
    x = jax.lax.batch_matmul(jnp.conj(Vh.swapaxes(-1,-2)),Uhb)

    if expand:
        x = x[...,0]

    return x


def l2_distance(X,Y):
    l2_dist = jnp.linalg.norm(X, axis=-1)**2 - 2 * np.conj(X) @ Y.T + jnp.linalg.norm(Y, axis=-1)**2
    return l2_dist.real
