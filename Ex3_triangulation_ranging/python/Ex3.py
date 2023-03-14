import numpy as np
import matplotlib.pyplot as plt
from matplotlib import checkdep_usetex
from scipy import stats

if checkdep_usetex(True):
    # Nicer plot fonts if available, if you have plotting errors
    plt.rcParams.update({"text.usetex": True,
                         "font.family": "serif",
                         "font.serif": ["Computer Modern Roman"],
                         "text.latex.preamble": r'\usepackage{amssymb}'})
plt.style.use('ggplot')


class LaserRanger:
    sigma_x = 0.0
    # Construct the laser ranger as a simple class

    def __init__(self, L, f):
        # Set the sensor constant parameters
        self.L = L
        self.f = f

    def range(self, x):
        # Q1a Measurement model
        # Return the range D as a function of the measured point on the sensor
        # We will define the return value of -1 in case of an error

        D = self.f*self.L/x

        return D

    def set_uncertainty(self, sigma_x):
        self.sigma_x = sigma_x

    def range_uncertainty(self, x):
        # Q2a - Standard deviation of the distance measurement
        sigma_D = self.f*self.L*self.sigma_x/(x**2)

        return sigma_D


# Create a sensor
laser = LaserRanger(L=1.0, f=0.5)

# Q1b
xx = np.linspace(0.05, 1.0, 50)
DD = [laser.range(xi) for xi in xx]

fh1, ah1 = plt.subplots()
ah1.plot(DD, xx, '.-')
ah1.set_xlabel('Range $D$')
ah1.set_ylabel('PSD measurement location $x$')
ah1.set_title('Laser ranging')

# Q2
laser.set_uncertainty(0.1)
sDD = [laser.range_uncertainty(xi) for xi in xx]

fh2, ah2 = plt.subplots()
ah2.plot(DD, sDD, '.-')
ah2.set_yscale('log')
ah2.set_xlabel('Range $D$')
ah2.set_ylabel('Measurement std $\sigma_D$')
ah2.set_title('Laser ranging with noise, $\sigma_x$={0:0.1f}'.format(laser.sigma_x))


# Measurement likelihood density plot - this is the likelihood of returning a
# certain D measurement given an actual x value
ylim = [0, DD[0]*1.5]    # Plot the densities for 0 to 1.5*maximum distance
n_density = 300
pD = np.linspace(ylim[0], ylim[1], n_density)

full_density = np.zeros((n_density, len(xx)))
for i, (xi, Di, sDi) in enumerate(zip(xx, DD, sDD)):
    full_density[:, i] = stats.norm.pdf(pD, loc=Di, scale=sDi)

fh3, ah3 = plt.subplots()
hp = ah3.imshow(full_density, origin='lower', extent=[xx[0], xx[-1], ylim[0], ylim[-1]], aspect='auto')
hD, = ah3.plot(xx, DD, '--', linewidth=0.5)
ah3.grid(False)
plt.colorbar(hp)
if checkdep_usetex(True):
    ah3.legend([hD], [r'$\mu_D = \mathbb{E}(D | x)$'])
else:
    ah3.legend([hD], [r'$\mu_D = E(D | x)$'])
ah3.set_xlabel('True $x$')
ah3.set_ylabel('Measured distance $D$')
ah3.set_title('Probability density, $p(D | x), \sigma_x={0:0.1f}$'.format(laser.sigma_x))

plt.show()