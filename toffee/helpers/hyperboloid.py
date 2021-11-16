import numpy as np 

def hyperboloid(x,y,z,mic1,mic2,delta):
    return np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y) + 
        np.square(mic1[2] - z)) - np.sqrt(np.square(mic2[0] - x) + 
        np.square(mic2[1] - y) + np.square(mic2[2] - z)) - delta

# delta +ve, mic1 closer than mic2
def hyperboloid_gradient(x,y,z,mic1,mic2,delta):
    denom1 = np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y) + np.square(mic1[2] - z))
    denom2 = np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y) + np.square(mic2[2] - z))
    x_grad = 2 * hyperboloid(x,y,z,mic1,mic2,delta) * ((x-mic1[0]) / denom1  -  (x-mic2[0]) / denom2)
    y_grad = 2 * hyperboloid(x,y,z,mic1,mic2,delta) * ((y-mic1[1]) / denom1  -  (y-mic2[1]) / denom2)
    z_grad = 2 * hyperboloid(x,y,z,mic1,mic2,delta) * ((z-mic1[2]) / denom1  -  (z-mic2[2]) / denom2)
    return np.array([x_grad, y_grad, z_grad])

def hyperbola(x,y,mic1,mic2,delta):
    return np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y)) - np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y)) - delta

def hyperbola_gradient(x,y,mic1,mic2,delta):
    x_grad = 2 * hyperbola(x,y,mic1,mic2,delta) * ((x-mic1[0]) / np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y))  -  (x-mic2[0]) / np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y)))
    y_grad = 2 * hyperbola(x,y,mic1,mic2,delta) * ((y-mic1[1]) / np.sqrt(np.square(mic1[0] - x) + np.square(mic1[1] - y))  -  (y-mic2[1]) / np.sqrt(np.square(mic2[0] - x) + np.square(mic2[1] - y)))
    return np.array([x_grad, y_grad])

# print(hyperbola_gradient(1, -1, [1,2], [1, -2], 2))


if __name__ == "__main__":
    # run grad descent on a known hyperbola, see what happens
    count = 0
    curr_guess = np.array([1.,1.])
    print(curr_guess)
    x = curr_guess[0]
    y = curr_guess[1]
    reg = 0.01
    alpha = 0.05
    grad = hyperbola_gradient(x, y, np.array([1.,2.]), np.array([1.,0.]), 1) + hyperbola_gradient(x, y, np.array([1.,2.]), np.array([0.,0.]), 1.0)
    while np.linalg.norm(grad + reg * curr_guess) >= 0.03 and count <= 3000:
        x = curr_guess[0]
        y = curr_guess[1]
        grad = hyperbola_gradient(x, y, np.array([1.,2.]), np.array([1.,0.]), 1) + hyperbola_gradient(x, y, np.array([1.,2.]), np.array([0.,0.]), 1)
        print(np.linalg.norm(grad + reg * curr_guess))
        curr_guess = curr_guess - alpha * (grad + curr_guess * reg)
        count += 1
    print(curr_guess)