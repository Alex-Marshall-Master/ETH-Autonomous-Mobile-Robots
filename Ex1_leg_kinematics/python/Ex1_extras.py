import numpy as np
import matplotlib.pyplot as plt
from matplotlib import checkdep_usetex
from matplotlib.animation import FuncAnimation

if checkdep_usetex(True):
    # Nicer plot fonts if available
    plt.rcParams.update({"text.usetex": True,
                         "font.family": "serif",
                         "font.serif": ["Computer Modern Roman"]})


def J_check(J, r, q, delta=1e-8):
    # Simple numerical difference Jacobian checker
    # Inputs J and r must be functions that return the Jacobian and position as
    # a function of the configuration respectively
    J_fun = J(q)
    J_num = np.zeros(J_fun.shape)

    for i in range(J_fun.shape[1]):
        qd = q.copy()
        qd[i] = qd[i] + delta
        J_num[:, [i]] = (r(qd) - r(q))/delta

    max_diff = abs(J_fun-J_num).max()

    if max_diff < 1e-6:
        print('Jacobian check passed, max difference = {0}'.format(max_diff))
        return True
    else:
        print('Jacobian check failed, max difference = {0}'.format(max_diff))
        print('Functional Jacobian: {0}'.format(J_fun))
        print('Numerical Jacobian: {0}'.format(J_num))
        return False


class TrajectoryPlotter:

    def __init__(self, timeArr, qArr, rArr, rGoalArr):

        self.timeArr = timeArr
        self.qArr = qArr
        self.rArr = rArr
        self.rGoalArr = rGoalArr

        self.rKArr = self.r_B3_inB(qArr)
        self.rHArr = self.r_B2_inB(qArr)

        self.h_fig, self.h_ax = self._create_axes()

    def r_B3_inB(self, q):
        return np.array([-np.sin(q[1]),
                         np.sin(q[0])*(np.cos(q[1]) + 1) + 1,
                         -np.cos(q[0])*(np.cos(q[1]) + 1)])

    def r_B2_inB(self, q):
        return np.array([np.zeros_like(q[0]),
                         np.sin(q[0]) + 1,
                         -np.cos(q[0])])

    def _create_axes(self):
        h_fig, h_ax = plt.subplots(1, 2)
        h_fig.set_size_inches(10, 5.5)
        h_ax[0].axis('equal')

        h_ax[0].set_xlabel('$x$ axis [m]')
        h_ax[0].set_ylabel('$z$ axis [m]')
        h_ax[0].set_title('Trajectory following')

        h_ax[1].set_xlabel('Time [s]')
        h_ax[1].set_ylabel('Position [m]')
        h_ax[1].set_title('Trajectory following')

        h_ax[0].set_xlim([-0.7, 1])
        h_ax[0].set_ylim([self.rGoalArr[2].min()-0.1, -0.50])

        return h_fig, h_ax

    def plot(self):

        pA, = self.h_ax[0].plot(self.rArr[0], self.rArr[2], 'r')
        pG, = self.h_ax[0].plot(self.rGoalArr[0], self.rGoalArr[2],'b')
        pAA, = self.h_ax[0].plot(self.rArr[0, 0], self.rArr[2, 0], 'ro')
        pGG, = self.h_ax[0].plot(self.rGoalArr[0, 0], self.rGoalArr[2, 0], 'bo');
        pL1, = self.h_ax[0].plot([self.rKArr[0,0], self.rArr[0,0]], [self.rKArr[2,0], self.rArr[2,0]], 'k-.')
        pL2, = self.h_ax[0].plot([self.rHArr[0,0], self.rKArr[0,0]], [self.rHArr[2,0], self.rKArr[2,0]],'k-.')
        pL3, = self.h_ax[0].plot([0, self.rHArr[0,0]],[0,self.rHArr[2,0]],'k-.')
        self.h_ax[0].legend([pG, pA, pL1], ['Target position','Actual position','Leg configuration'])

        pZa, = self.h_ax[1].plot(self.timeArr, self.rArr[2,:], 'r')
        pZt, = self.h_ax[1].plot(self.timeArr, self.rGoalArr[2,:], 'b')
        pR1, = self.h_ax[1].plot(self.timeArr[0], self.rArr[2, 0], 'r.')
        pR2, = self.h_ax[1].plot(self.timeArr[0], self.rGoalArr[2, 0], 'b.')
        self.h_ax[1].legend([pZt, pZa], ['$z$-target [m]', '$z$-position [m]'])

        self.artists = [pAA, pGG, pL1, pL2, pL3, pR1, pR2]
        return self.artists

    def _animate(self, i):
        assert i < len(self.timeArr)

        self.artists[0].set_data(self.rArr[0, :i], self.rArr[2, :i])
        self.artists[1].set_data(self.rGoalArr[0, :i], self.rGoalArr[2, :i])
        self.artists[2].set_data([self.rKArr[0, i], self.rArr[0, i]],
                                 [self.rKArr[2, i], self.rArr[2, i]])
        self.artists[3].set_data([self.rHArr[0, i], self.rKArr[0, i]],
                                 [self.rHArr[2, i], self.rKArr[2, i]])
        self.artists[4].set_data([0, self.rHArr[0, i]],
                                 [0, self.rHArr[2, i]])
        self.artists[5].set_data(self.timeArr[:i], self.rArr[2, :i])
        self.artists[6].set_data(self.timeArr[:i], self.rGoalArr[2, :i])

        return self.artists

    def animation(self):
        animation = FuncAnimation(self.h_fig, self._animate,
                                  init_func=self.plot, frames=len(self.timeArr),
                                  interval=20, blit=True, repeat=True)
        return animation
