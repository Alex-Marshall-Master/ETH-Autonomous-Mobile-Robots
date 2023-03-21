import numpy as np
import matplotlib.pyplot as plt


# Autonomous Mobile Robots - Exercise 4 (Image Formation and Stereo Vision)

def point_cloud_plot(xyz, name):
    fh = plt.figure(num=name)
    axh = fh.add_subplot(projection='3d')
    hp = axh.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2])
    axh.set_box_aspect((np.ptp(xyz[:, 0]),
                        np.ptp(xyz[:, 1]),
                        np.ptp(xyz[:, 2])))
    return fh, axh, hp


# Q1.7 Camera Calibration and Image Formation.
# In this exercise we will use the extrinsic and intrinsic calibration
# matrices, which were computed in the previous exercises, to create an
# image from a given 3D structure.

# For visualizing, we will use a 3D point cloud of the so called 'Utah
# Teapot', which is a well known 3D test model.
# https://en.wikipedia.org/wiki/Utah_teapot
# Take a look at the plot to get an idea what to expect in the upcoming
# questions.
xyzPoints = np.genfromtxt('Ex4_image_formation_stereo_vision\\python\\teapot.xyz', delimiter=',')
f1, ax1, h_pot = point_cloud_plot(xyzPoints, 'Utah Teapot (3D)')


# In the following, we will only consider one half (-Y_w) of the
# teapot. This is to approximate the fact that the front part will occlude
# the back part when seen from the camera.
# Note: In reality, we would have to consider the camera's exact viewing
# point and perform occlusion detection when rendering the image.
# However, for the sake of simplicity we will make the above approximation.

# Only consider front half.
xyzPoints = xyzPoints[xyzPoints[:, 1] < 0, :]


# First, we will define the parameters of the intrinsic and extrinsic
# calibration.
img_size_x = 640   # Image size (px) along the horizontal (Xc/u) direction.
img_size_y = 320   # Image size (px) along the vertical (Yc/v) direction.
fov_x_rad = 60.0*np.pi/180    # Field of View (rad) along horizontal direction.
fov_y_rad = 45.0*np.pi/180    # Field of View (rad) along vertical direction.

# TODO: Fill in the missing values from the values computed in Q1-4.
alpha_x = img_size_x*0.5/np.tan(fov_x_rad*0.5)  # Focal length in pixels along camera's X axis (horizontal)
alpha_y = img_size_y*0.5/np.tan(fov_y_rad*0.5)  # Focal length in pixels along camera's Y axis (vertical)
u0 = img_size_x/2   # Center of image plane in pixels along horizontal direction
v0 = img_size_y/2   # Center of image plane in pixels along vertical direction
K = np.array([[alpha_x, 0, u0], [0,alpha_y,v0], [0,0,1]])    # 3x3 intrinsic camera calibration matrix
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])    # 3x3 rotation matrix: World -> Camera
t = np.array([[0], [1.5], [5.0]])    # 3x1 translation vector: World -> Camera


# TODO: Implement the function "project_to_image"
def project_to_image(K, R, t, Pw, img_size_x, img_size_y):
    Pc = R @ Pw + t
    Pc_pixel = K @ Pc

    u = Pc_pixel[0, 0]/Pc_pixel[2, 0]
    v = Pc_pixel[1, 0]/Pc_pixel[2, 0]

    if u > img_size_x or u < 0:
        u = np.nan
    if  v > img_size_y or v < 0:
        v = np.nan

    return u, v


# Now, we will render the image of the teapot by projecting each point
# of the 3D point cloud on to the 2D image plane and visualize it.
u_list = []
v_list = []

for i in range(xyzPoints.shape[0]):
    u, v = project_to_image(K, R, t, xyzPoints[[i],:].T, img_size_x, img_size_y)
    if not(np.isnan(u) or np.isnan(v)):
        # Only consider valid projections.
        u_list.append(u)
        v_list.append(v)

# Let's look what the image looks like.
fig2, ax2 = plt.subplots(num='Image of Utah Teapot')
ax2.scatter(u_list, v_list)
ax2.set_aspect('equal')
ax2.set_xlabel('u [px]')
ax2.set_ylabel('v [px]')

# Do you see a sideview of the teapot?

# Note: In a real camera, there are infinitely many points from the entire
# scene which project to the image plane to create an image.
# However, in computer graphics we can only render a finite amount of 3D points.

# Play around with the extrinsic and intrinsic parameters of the camera.
# and see how it affects the created image.
# Answer the following questions.
# 1.) What happens if we translate the camera location along the Yw axis?
# 2.) How could you get a close up view of the teapot WITHOUT moving the
# camera?
# 3.) How would you get a top down view of the teapot?

## Q2.4 Stereo Triangulation
# In this exercise, we will look at how a 3D structure can be
# reconstructed with a stereo camera setup.
# The stereo setup is the same as introduced in Q2-1.

# Parameters (we will be slightly modifying the setup from Q1-7).
baseline = 1.0      # Offset between left and right c
offset_y = 3.5
offset_z = 10.0     # How far away the stereo setup is placed from the world frame (along Z).
R_left = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
t_left = np.array([[baseline/2.0], [offset_y], [offset_z]])
R_right = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
t_right = np.array([[-baseline/2.0], [offset_y], [offset_z]])

# Note: In reality, we only have access to the 2 images of the same scene
# and we need to perform correspondence search in order to create a
# disparity map.
# However, for educational reasons we will create the disparity map
# ourselves by using the ground truth information we have from the given
# point cloud and stereo setup.

# Project the teapot into the two cameras.
u_left_list = []
v_left_list = []
u_right_list = []
v_right_list = []
disparities = []

for i in range(xyzPoints.shape[0]):
    u_left, v_left = project_to_image(K, R_left, t_left, xyzPoints[[i],:].T, img_size_x, img_size_y)
    u_right, v_right = project_to_image(K, R_right, t_right, xyzPoints[[i],:].T, img_size_x, img_size_y)
    if not(np.isnan(u_left) or np.isnan(v_left) or np.isnan(u_right) or np.isnan(v_right)):
        u_left_list.append(u_left)
        v_left_list.append(v_left)
        u_right_list.append(u_right)
        v_right_list.append(v_right)
        d =  u_left - u_right
        disparities.append(d)


# Let's now look at how the two images look like and what the respective
# disparity values are.

# Plot the projected image of the teapot.
ss1 = 4 # Subsampling for visualizing teapot image.
ss2 = 20 # Subsampling for visualizing disparities.
f3, a3 = plt.subplots(num='Disparity Map')
hl = a3.scatter(u_left_list[1::ss1], v_left_list[1::ss1], marker="o", facecolors='none', edgecolors='r', s=20) 
hr = a3.scatter(u_right_list[1::ss1], v_right_list[1::ss1], marker="o", facecolors='none', edgecolors='b', s=20)   
# Highlight the disparity for a small subset of points.
hl = a3.scatter(u_left_list[1::ss2], v_left_list[1::ss2], marker="o", facecolors='r', edgecolors='r', s=50) 
hr = a3.scatter(u_right_list[1::ss2], v_right_list[1::ss2], marker="o", facecolors='b', edgecolors='b', s=50)   
hd = a3.quiver(u_left_list[1::ss2], v_left_list[1::ss2], -np.array(disparities[1::ss2]), np.zeros_like(disparities[1::ss2]), width=0.0015, scale_units='xy', scale=1.)
a3.set_aspect('equal')
a3.set_xlabel('u [px]')
a3.set_ylabel('v [px]')
a3.legend([hl, hr, hd], ['Left Image','Right Image', 'Disparity'])

# TODO: Play around with the camera parameters.
# * What happens if you make the baseline very small/large?

# Now, we will re-triangulate the 3D structure by knowing the disparity map
# and the stereo configuration.
# Remember again that in reality we will only have access to these!

# TODO: Please implement the triangulation function derived in Q2-1.
# Use Python's 'lambda functions'.

# Noted: perspective only moves in x direction, so use fx (focal length in x direction)
z_triangulated = lambda disparity: alpha_x*baseline/disparity

# We will recover the 3D structure from the disparity map.

# Add white Gaussian noise, inspired by
# https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
target_snr_db = 50
noise_avg_db = 10 * np.log10(np.mean(disparities)) - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
disparity_noisy = np.array(disparities) + np.random.normal(0, np.sqrt(noise_avg_watts), len(disparities))

# Triangulate.
triangulated_pc = []
for i in range(len(u_left_list)):

    # Triangulate the depth 'Z'.
    z_triang = z_triangulated(disparity_noisy[i])
    # Also recover X and Y.
    x_triang = ((u_left_list[i] - u0)/alpha_x)*z_triang
    y_triang = ((v_left_list[i] - v0)/alpha_y)*z_triang
    triang_pt = [x_triang, y_triang, z_triang]
    triangulated_pc.append(triang_pt)

# Visualize the triangulated point cloud.
triangulated_pc = np.array(triangulated_pc)
f4, ax4, hp4 = point_cloud_plot(triangulated_pc, 'Utah Teapot (3D) Triangulated')

plt.show()
