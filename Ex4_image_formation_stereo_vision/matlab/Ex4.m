%% Autonomous Mobile Robots - Exercise 4 (Image Formation and Stereo Vision)
close all;
clear all;
clc;

%% Q1.7 Camera Calibration and Image Formation.
% In this exercise we will use the extrinsic and intrinsic calibration
% matrices, which were computed in the previous exercises, to create an
% image from a given 3D structure.

% For visualizing, we will use a 3D point cloud of the so called 'Utah
% Teapot', which is a well known 3D test model.
% https://en.wikipedia.org/wiki/Utah_teapot
% Take a look at the plot to get an idea what to expect in the upcoming
% questions.
load('xyzPoints');
ptCloud = pointCloud(xyzPoints);
figure('Name','Utah Teapot (3D)')
pcshow(ptCloud)

% In the following, we will only one half (-Y_w) of the
% teapot. This is to approximate the fact that the front part will occlude
% the back part when seen from the camera.
% Note: In reality, we would have to consider the camera's exact viewing
% point and perform occlusion detection when rendering the image.
% However, for the sake of simplicity we will make the above approximation.

% Only consider front half.
mug_sliced = [];
for i = 1:size(xyzPoints,1)
    pt = xyzPoints(i,:);
    if pt(2) < 0
        mug_sliced = [mug_sliced; pt];
    end
end
xyzPoints = mug_sliced;

% First, we will define the parameters of the intrinsic and extrinsic
% calibration.
img_size_x = 640;   % Image size (px) along the horizontal (Xc/u) direction.
img_size_y = 320;   % Image size (px) along the vertical (Yc/v) direction.
fov_x_rad = deg2rad(60);    % Field of View (rad) along horizontal direction.
fov_y_rad = deg2rad(45);    % Field of View (rad) along vertical direction.

% TODO: Fill in the missing values from the values computed in Q1-4.
alpha_x = img_size_x*0.5/tan(fov_x_rad*0.5);  % Focal length in pixels along camera's X axis (horizontal)
alpha_y = img_size_y*0.5/tan(fov_y_rad*0.5);  % Focal length in pixels along camera's Y axis (vertical)
u0 = img_size_x/2;   % Center of image plane in pixels along horizontal direction
v0 = img_size_y/2;   % Center of image plane in pixels along vertical direction
K = [alpha_x, 0, u0; 0,alpha_y,v0; 0,0,1];   % 3x3 intrinsic camera calibration matrix
R = [1, 0, 0; 0, 0, -1; 0, 1, 0];    % 3x3 rotation matrix: World -> Camera
t = [0; 1.5; 5.0];    % 3x1 translation vector: World -> Camera

% TODO: Implement the function "projectToImage" at the bottom of this file.

% Now, we will render the image of the teapot by projecting each point 
% of the 3D point cloud on to the 2D image plane and visualize it.
u_list = [];
v_list = [];

for i = 1:size(xyzPoints,1)
    [u, v] = projectToImage(K, R, t, xyzPoints(i,:)', img_size_x, img_size_y);
    if not(isnan(u) || isnan(v))
        % Only consider valid projections.
        u_list = [u_list, u];
        v_list = [v_list, v];
    end
end

% Let's look what the image looks like.
figure('Name','Image of Utah Teapot')
scatter(u_list, v_list);
axis equal
xlabel('u [px]');
ylabel('v [px]');

% Do you see a sideview of the teapot?

% Note: In a real camera, there are infinitely many points from the entire
% scene which project to the image plane to create an image.
% However, in computer graphics we can only render a finite amount of 3D points.

% Play around with the extrinsic and intrinsic parameters of the camera.
% and see how it affects the created image.
% TODO: Answer the following questions.
% 1.) What happens if we translate the camera location along the Yw axis?
% 2.) How could you get a close up view of the teapot WITHOUT moving the
% camera?
% 3.) How would you get a top down view of the teapot?

%% Q2.4 Stereo Triangulation
% In this exercise, we will look at how a 3D structure can be 
% reconstructed with a stereo camera setup.
% The stereo setup is the same as introduced in Q2-1.

% Parameters (we will be slightly modifying the setup from Q1-7).
baseline = 1.0; % Offset between left and right c
offset_y = 3.5;
offset_z = 10.0; % How far away the stereo setup is placed from the world frame (along Z).
R_left = [1, 0, 0; 0, 0, -1; 0, 1, 0];
t_left = [baseline/2.0; offset_y; offset_z];
R_right = [1, 0, 0; 0, 0, -1; 0, 1, 0];
t_right = [-baseline/2.0; offset_y; offset_z];

% Note: In reality, we only have access to the 2 images of the same scene
% and we need to perform correspondence search in order to create a
% disparity map. 
% However, for educational reasons we will create the disparity map
% ourselves by using the ground truth information we have from the given
% point cloud and stereo setup.

% Project the teapot into the two cameras.
u_left_list = [];
v_left_list = [];
u_right_list = [];
v_right_list = [];
disparities = [];

for i = 1:size(xyzPoints,1)
    [u_left, v_left] = projectToImage(K, R_left, t_left, xyzPoints(i,:)', img_size_x, img_size_y);
    [u_right, v_right] = projectToImage(K, R_right, t_right, xyzPoints(i,:)', img_size_x, img_size_y);
    if not(isnan(u_left) || isnan(v_left) || isnan(u_right) || isnan(v_right))
        u_left_list = [u_left_list, u_left];
        v_left_list = [v_left_list, v_left];
        u_right_list = [u_right_list, u_right];
        v_right_list = [v_right_list, v_right];
        d =  u_left - u_right;
        disparities = [disparities; d];
    end
end

% Let's now look at how the two images look like and what the respective
% disparity values are.

% Plot the projected image of the teapot.
ss1 = 4; % Subsampling for visualizing teapot image.
ss2 = 20; % Subsampling for visualizing disparities.
figure('Name','Disparity Map')
scatter(u_left_list(1:ss1:end), v_left_list(1:ss1:end), 20, 'ro')
hold on;
scatter(u_right_list(1:ss1:end), v_right_list(1:ss1:end), 20, 'bo')
hold on;
% Highlight the disparity for a small subset of points.
scatter(u_left_list(1:ss2:end), v_left_list(1:ss2:end), 50, 'ro', '+', 'filled')
hold on;
scatter(u_right_list(1:ss2:end), v_right_list(1:ss2:end), 50, 'bo', '+', 'filled')
hold on;
quiver(u_left_list(1:ss2:end)', v_left_list(1:ss2:end)',-disparities(1:ss2:end),0*disparities(1:ss2:end), 'AutoScale','off', 'LineWidth',1)
axis equal
xlabel('u [px]');
ylabel('v [px]');
legend('Left Image','Right Image', '', '', 'Disparity');

% TODO: Play around with the camera parameters.
% * What happens if you make the baseline very small/large?

% Now, we will re-triangulate the 3D structure by knowing the disparity map
% and the stereo configuration.
% Remember again that in reality we will only have access to these!

% TODO: Please implement the triangulation function derived in Q2-1.
% Use MATLAB's anonymous functions.
z_triangulated = @(disparity) alpha_x*baseline/disparity;

% We will recover the 3D structure from the disparity map.
disparity_noisy = awgn(disparities,50,'measured'); % add some noise.
% Triangulate.
triangulated_pc = [];
for i = 1:size(u_left_list,2)
    % Triangulate the depth 'Z'.
    z_triang = z_triangulated(disparity_noisy(i));
    % Also recover X and Y.
    x_triang = ((u_left_list(i) - u0)/alpha_x)*z_triang;
    y_triang = ((v_left_list(i) - v0)/alpha_y)*z_triang;
    triang_pt = [x_triang; y_triang; z_triang];
    triangulated_pc = [triangulated_pc; triang_pt'];
end

% Visualize the triangulated point cloud.
ptCloud2 = pointCloud(triangulated_pc);
figure('Name','Utah Teapot (3D) Triangulated')
pcshow(ptCloud2)


%% Function Definitions.
% TODO: Implement the projectToImage function.
function [u, v] = projectToImage(K, R, t, Pw, img_size_x, img_size_y)
    Pc = R * Pw + t;
    Pc_pixel = K * Pc;

    u = Pc_pixel(1)/Pc_pixel(3);
    v = Pc_pixel(2)/Pc_pixel(3);

    if u > img_size_x || u < 0
        u = nan;
    end
    if  v > img_size_y || v < 0
        v = nan;
    end
end
