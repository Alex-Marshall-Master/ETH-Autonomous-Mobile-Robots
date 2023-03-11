% Note that there are several possible solutions to this 
% depending on your frame definitions. However, the stacked
% wheel equations should still hold


% Wheel 1, the far right wheel
alpha1 = 0;
beta1 = 0;
ell1 = 0.5;

% Wheel 2, the top left wheel
alpha2 = 2.0*pi/3.0;
beta2 = 0;
ell2 = 0.5;
      
% Wheel 3, the bottom left wheel
alpha3 = 4.0*pi/3.0;
beta3 = 0;
ell3 = 0.5;

% The wheel radius
r = 0.1;
  
% Build the equations for each wheel by plugging in the parameters
J1 = [sin(alpha1+beta1), -cos(alpha1+beta1), -ell1*cos(beta1)];
J2 = [sin(alpha2+beta2), -cos(alpha2+beta2), -ell2*cos(beta2)];
J3 = [sin(alpha3+beta3), -cos(alpha3+beta3), -ell3*cos(beta3)];
  
% Stack the wheel equations
J = [J1;J2;J3];
R = eye(3)*r;

% Compute the forward differential kinematics matrix, F
F = J\R;

%% Try changing the wheel speeds to see what motions the robot does.
numSeconds=10;
dt = 0.1;

% The speed of the first wheel (rad/s)
phiDot1 = 1.0*ones(1, numSeconds/dt);
% The speed of the second wheel (rad/s)
phiDot2 = 0.5*ones(1, numSeconds/dt);
% The speed of the third wheel (rad/s)
phiDot3 = 0.25*ones(1, numSeconds/dt);
phiDot = [phiDot1; phiDot2; phiDot3];

% Ex3 Inverse kinematics
F_inv = R\J;
ones_list = ones(1, numSeconds/dt);
zeros_list = zeros(1, numSeconds/dt);
t_list = linspace(0,numSeconds,numSeconds/dt);
% Stationary rotation (1 full rotations in 10 seconds, i.e. 0.1Hz)
stateDot = [zeros_list; zeros_list; 0.5*ones_list];
phiDot = F_inv * stateDot;

% Linear motion in R_X
stateDot = [0.1*ones_list; zeros_list; zeros_list];
phiDot = F_inv * stateDot;

% In a circle (no rotation)
stateDot = [0.5*cos(t_list); 0.5*sin(t_list); zeros_list];
phiDot = F_inv * stateDot;

% BONUS: In a circle + constant rotation
stateDot = [0.5*cos(t_list); 0.5*sin(t_list); -0.5*ones_list];
phiDot = F_inv * stateDot;



plotOmnibot(F, phiDot, dt);
