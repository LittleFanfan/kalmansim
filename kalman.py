import numpy as np
from bokeh.client import push_session
from bokeh.plotting import figure, curdoc

# True system parameters
k = 1.0
m = 1.0
b = 1.0
A = np.array([[0.0, 1.0],
            [-k/m, -b/m]])
B = np.array([[0.0],
            [1.0/m]])

# Model parameters
k_hat = 0.9
m_hat = 1.4
b_hat = 0.9
A_hat = np.array([[0.0, 1.0],
                [-k_hat/m_hat, -b_hat/m_hat]])
B_hat = np.array([[0.0],[1.0/m_hat]])
C = np.array([[1.0, 0.0]])
# # OBSV Check
# obsv = np.vstack((C_hat, np.dot(C_hat,A_hat)))
# obsv_rank = np.linalg.matrix_rank(obsv)

# Sensor Covariance
sigmaR = 0.2
R = np.array([sigmaR**2])
# Process Noise
sigmaQ = 0.02
Q = np.array([sigmaQ**2])

# time and control vectors
dt = 0.1
duration = 100.0
t = np.linspace(0.0,duration, duration/dt)
u = np.sin(t/5.0)
# u = np.ones(t.shape)

# Initial conditions
x0 = 0.0
v0 = 0.0
x = np.array([[x0],[v0]])
x_hat = x = np.array([[x0],[v0]])
P = np.eye(2);

# Initialize data storage arrays
output = np.array([[],[]])
noisy_output = np.array([[],[]])
output_hat = np.array([[],[]])

# Run integration
for sample, T in enumerate(t):
    # Real System
    xdot = A.dot(x) + B.dot(u[sample])
    x = x + xdot*dt
    output = np.hstack((output, x))

    # Noisy Measurement
    try:
        noisy_x = x + np.array([[np.random.normal(scale=sigmaR)],
                            [np.random.normal(scale=sigmaR)]])
    except:
        noisy_x = x
    noisy_output = np.hstack((noisy_output, noisy_x))
    y = C.dot(noisy_x)

    # Model Prediction
    xdot_hat = A_hat.dot(x_hat) + B_hat.dot(u[sample])
    x_hat = x_hat + xdot_hat*dt
    P = A_hat.dot(P).dot(A_hat.T) + Q

    # Estimate Update
    K = P.dot(C.T).dot(np.linalg.inv(C.dot(P).dot(C.T) + R))
    x_hat = x_hat + K.dot(y - C.dot(x_hat))
    P = P - K.dot(C).dot(P)

    # Output
    output_hat = np.hstack((output_hat, x_hat))


#start bokeh plotting session
session = push_session(curdoc())


# Plots
f = figure()
f.line(t[:], noisy_output[0,:], line_color='green')
f.line(t[:], output_hat[0,:], line_color='blue')
f.line(t[:], output[0,:], line_color='red')
show(f)
