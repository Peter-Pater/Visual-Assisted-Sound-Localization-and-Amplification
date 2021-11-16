import numpy as np
import tqdm
from scipy.fft import fft, fftfreq, ifft

def grad(W, x):
    x_h = x.H
    R_xx = np.matmul(x, x_h)

    y = np.matmul(W, x)
    y_h = y.H
    R_yy = np.matmul(y, y_h)
    #R_yy = np.matmul(np.matmul(W, R_xx), W.H)
    
    A = np.asmatrix(np.linalg.lstsq(W, np.identity(2), rcond = 1)[0])

    E = R_yy - np.diag(np.diag(R_yy))

    dJ1 = 4*np.matmul(np.matmul(E, W), R_xx)
    dJ2 = 2*np.matmul((np.matmul(W, A) - np.identity(W.shape[0])), A.H)

    if (np.linalg.norm(R_xx)) == 0:
        print("HI")
        alpha = 1
    else:
        alpha = (np.linalg.norm(R_xx))**-2
    grad = alpha * dJ1 + dJ2
  
    return grad

def grad_descent(W_init, x, mu, reg = 0.01):
    W = W_init
    count = 0
    g = grad(W, x)
    while np.linalg.norm(g) >= 0.00005:
        g = grad(W, x)
        W = (1-mu*reg) * W - mu*g
    return W

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def grad_descent_all_frames(data, frames):
    final_data = []
    source1 = []
    source2 = []
    for i in tqdm.tqdm(range(0, data.shape[1] - frames, frames//4)):
        frame_data = data[:, i:i+(frames//4)]
        x = fft(frame_data)
        if i == 0:
            W_init = np.random.rand(2*data.shape[0]) + np.random.rand(2*data.shape[0]) * 1j
            W_init = W_init.reshape((2, data.shape[0]))

            x = np.asmatrix(x)
            W_init = np.asmatrix(W_init)
        else:
            x = np.asmatrix(x)
            W_init = np.asmatrix(W_final)
        
        W_final = grad_descent(W_init, x, 0.01)
        # W_real = np.real(W_final)
        # amp_sum = np.sum(np.abs(W_real))
        # W_final = W_final / amp_sum
        
        W_1 = W_final[0]
        W_2 = W_final[1]
        W_1real = np.real(W_1)
        W_2real = np.real(W_2)
        
        c1 = np.sqrt(1./np.sum(np.square(W_1real)))
        c2 = np.sqrt(1./np.sum(np.square(W_2real)))

        W_final = np.array([W_1 * c1, W_2 * c2])
        W_final = np.asmatrix(W_final)
        
        # W_final = np.asmatrix(np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0]]))

        frame = data[:, i:i+frames//4]
        interval_x = fft(frame)
        interval_x = np.asmatrix(interval_x)

        y_freq = W_final * interval_x
        y = ifft(y_freq)
        source = (np.real(y))

        A = np.asmatrix(np.linalg.lstsq(W_final, np.identity(2), rcond = None)[0])
        
        A1 = np.copy(A)
        A2 = np.copy(A)
        A1[:, 1] = np.zeros([1,data.shape[0]])
        A2[:, 0] = np.zeros([1,data.shape[0]])
        source1.append(np.real(np.matmul(A1, source)))
        source2.append(np.real(np.matmul(A2, source)))
        
        final_data.append(source)
    return W_final, A, final_data, source1, source2


if __name__ == "__main__":
    W = np.asmatrix(np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0]]))
    A = np.asmatrix(np.linalg.lstsq(W, np.identity(2), rcond = 1)[0])
    print(W, A)
    print(np.matmul(W, A))


    y = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(np.diag(np.diag(y)))