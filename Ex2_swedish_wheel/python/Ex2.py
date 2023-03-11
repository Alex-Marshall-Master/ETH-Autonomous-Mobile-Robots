import numpy as np
import matplotlib.pyplot as plt
from plotOmnibot import OmnibotPlotter

# Wheel 1, the far right wheel
alpha1 = 0
beta1 = 0
ell1 = 0.5

# Wheel 2, the top left wheel
alpha2 = 2.0*np.pi/3.0
beta2 = 0
ell2 = 0.5

# Wheel 3, the bottom left wheel
alpha3 = 4.0*np.pi/3.0
beta3 = 0
ell3 = 0.5

# The wheel radius
r = 0.1

# Build the equations for each wheel by plugging in the parameters (1x3 arrays)
J1 = np.array([np.sin(alpha1+beta1), -np.cos(alpha1+beta1), -ell1*np.cos(beta1)])
J2 = np.array([np.sin(alpha2+beta2), -np.cos(alpha2+beta2), -ell2*np.cos(beta2)])
J3 = np.array([np.sin(alpha3+beta3), -np.cos(alpha3+beta3), -ell3*np.cos(beta3)])


# Stack the wheel equations
J = np.array([J1, J2, J3])
R = np.eye(3)*r


# Ex2 Forward kinematics
# Compute the forward differential kinematics matrix, F
F = np.linalg.pinv(J) @ R

## Try changing the wheel speeds to see what motions the robot does
numSeconds = 10
dt = 0.1
tt = np.arange(0, numSeconds, dt)

# The speed of the first wheel (rad/s)
phiDot1 = 1.0*np.ones_like(tt)
# The speed of the second wheel (rad/s)
phiDot2 = 0.5*np.ones_like(tt)
# The speed of the third wheel (rad/s)
phiDot3 = 0.25*np.ones_like(tt)
phiDot = np.array([phiDot1, phiDot2, phiDot3])



# Ex3 Inverse kinematics
F_inv = np.linalg.pinv(R) @ J

# Stationary rotation (1 full rotations in 10 seconds, i.e. 0.1Hz)
stateDot= np.array([np.zeros_like(tt), np.zeros_like(tt), 0.1*np.ones_like(tt)])
phiDot = F_inv @ stateDot

# Linear motion in R_X
stateDot = np.array([0.1*np.ones_like(tt), np.zeros_like(tt), np.zeros_like(tt)])
phiDot = F_inv @ stateDot

# In a circle (no rotation)
stateDot = np.array([0.5*np.cos(tt), 0.5*np.sin(tt), np.zeros_like(tt)])
phiDot = F_inv @ stateDot

# BONUS: In a circle + constant rotation
stateDot = np.array([0.5*np.cos(tt), 0.5*np.sin(tt), -0.5*np.ones_like(tt)])
phiDot = F_inv @ stateDot

omni_animator = OmnibotPlotter(F, phiDot, dt)
an = omni_animator.animation()
plt.show()