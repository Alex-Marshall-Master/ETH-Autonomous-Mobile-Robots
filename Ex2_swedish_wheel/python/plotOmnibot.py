import numpy as np
import matplotlib.pyplot as plt
from matplotlib import checkdep_usetex
from matplotlib.animation import FuncAnimation

if checkdep_usetex(True):
    # Nicer plot fonts if available
    plt.rcParams.update({"text.usetex": True,
                         "font.family": "serif",
                         "font.serif": ["Computer Modern Roman"]})


class OmnibotPlotter:
    artists = []

    def __init__(self, F, phi_dot, dt, initial_state=[[0.0], [0.0], [0.0]]):

        self.F = F
        self.dt = dt
        self.phi_dot = phi_dot
        self.initial_state = initial_state

        self.full_time, self.full_state = self._forward_simulate()
        self.h_fig, self.h_ax = self._create_axes()

        # Define some local positions for plotting the robot axes
        # (these are in the robot frame)
        self._px = np.atleast_2d([0.1, 0]).T
        self._py = np.atleast_2d([0, 0.1]).T

    @staticmethod
    def _R(theta):
        # Rotation matrix for robot angle theta
        return np.array([[np.cos(theta), np.sin(theta), 0.0],
                         [-np.sin(theta), np.cos(theta), 0.0],
                         [0, 0, 1]])

    def _forward_simulate(self):
        steps = self.phi_dot.shape[1]

        # Save an array of the full state history
        full_state = np.zeros((3, steps+1))
        full_time = np.linspace(0, steps*self.dt, steps+1)
        full_state[:, [0]] = self.initial_state

        xi = full_state[:, [0]]
        R = self._R(xi[2, 0])

        for i in range(steps):
            # Propagate xi forward in time
            xi = xi + R.T @ self.F @ self.phi_dot[:, [i]] * self.dt
            R = self._R(xi[2, 0])
            full_state[:, [i+1]] = xi

        return full_time, full_state

    def _create_axes(self):
        h_fig = plt.figure()
        h_fig.set_size_inches(10, 5.5)

        h_ax = [h_fig.add_subplot(1, 2, 1), h_fig.add_subplot(3, 2, 2),
                h_fig.add_subplot(3, 2, 4), h_fig.add_subplot(3, 2, 6)]

        h_ax[0].axis('equal')
        h_ax[0].set_xlabel('$x$ [m]')
        h_ax[0].set_ylabel('$y$ [m]')
        h_ax[0].set_title('Omnidirectional robot')
        h_ax[0].set_xlim([self.full_state[0].min() - 0.5, self.full_state[0].max() + 0.5])
        h_ax[0].set_ylim([self.full_state[1].min() - 0.5, self.full_state[1].max() + 0.5])

        for ha, l, v in zip(h_ax[1:], [r'$x$', r'$y$', r'$\theta$'], self.full_state):
            ha.set_xlabel('Time [s]')
            ha.set_ylabel(l)
            ha.plot(self.full_time, v)
            ha.set_xlim([self.full_time[0], self.full_time[-1]])
            ha.set_ylim([v.min()-0.1, v.max()+0.1])

        return h_fig, h_ax

    def _get_robot_axis_points(self, i):
        tip_points = np.zeros((2, 3), dtype=float)
        tip_points[:, 0] = self.full_state[:2, i]
        Rxy = self._R(self.full_state[2, i])[:2, :2]
        tip_points[:, [1]] = Rxy.T @ self._px + tip_points[:, [0]]
        tip_points[:, [2]] = Rxy.T @ self._py + tip_points[:, [0]]
        return tip_points

    def _init(self):
        for h in self.artists:
            h.remove()

        tip_points = self._get_robot_axis_points(0)
        h_Rx, = self.h_ax[0].plot(tip_points[0, :2], tip_points[1, :2],'r-')
        h_Ry, = self.h_ax[0].plot(tip_points[0, [0, 2]], tip_points[1, [0, 2]], 'g-')
        h_Rp, = self.h_ax[0].plot([tip_points[0, 0]], [tip_points[0, 1]], 'k-')

        h_x, = self.h_ax[1].plot([self.full_time[0]], [self.full_state[0, 0]],
                                 'ro')
        h_y, = self.h_ax[2].plot([self.full_time[0]], [self.full_state[1, 0]],
                                 'ro')
        h_t, = self.h_ax[3].plot([self.full_time[0]], [self.full_state[2, 0]],
                                 'ro')
        self.artists = [h_Rx, h_Ry, h_Rp, h_x, h_y, h_t]
        return self.artists

    def _animate(self, i):
        assert i < len(self.full_time)

        tip_points = self._get_robot_axis_points(i)
        self.artists[0].set_data(tip_points[0, :2], tip_points[1, :2])
        self.artists[1].set_data(tip_points[0, [0,2]], tip_points[1, [0,2]])
        self.artists[2].set_data(self.full_state[0, :i], self.full_state[1, :i])

        self.artists[3].set_data([self.full_time[i]], [self.full_state[0, i]])
        self.artists[4].set_data([self.full_time[i]], [self.full_state[1, i]])
        self.artists[5].set_data([self.full_time[i]], [self.full_state[2, i]])

        return self.artists

    def animation(self):
        animation = FuncAnimation(self.h_fig, self._animate,
                                  init_func=self._init, frames=len(self.full_time),
                                  interval=int(1000.0*self.dt), blit=True, repeat=True)
        return animation