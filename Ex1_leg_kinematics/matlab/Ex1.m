%% Q1
% Write down the rotation matrices as functions of the angles
% alpha, beta, gamma using anonymous functions 
% https://www.mathworks.com/help/matlab/matlab_prog/anonymous-functions.html
% Hint: R_B1 = @(alpha) [1,0,0; 0, cos(alpha), -sin(alpha); ... ];

R_B1 = @(alpha) [1,0,0; 0, cos(alpha), -sin(alpha); 0, sin(alpha), cos(alpha)];
R_12 = @(beta)  [cos(beta), 0, sin(beta); 0,1,0; -sin(beta), 0, cos(beta)];
R_23 = @(gamma) [cos(gamma),0,sin(gamma); 0,1,0; -sin(gamma),0, cos(gamma)];



%% Q2
% Write down the 3x1 relative position vectors for link lengths l_i=1
r_3F_in3 = [0;0;-1];
r_23_in2 = [0;0;-1];
r_12_in1 = [0;0;-1];
r_B1_inB = [0;1;0];

% Write down the homogeneous transformations
% Hint: Can be created as compound matrices: [R_XY(gamma), r_XY_inX; 0 0 0 1];
H_23 = @(gamma) [R_23(gamma), r_23_in2; 0 0 0 1];
H_12 = @(beta)  [R_12(beta), r_12_in1; 0 0 0 1];
H_B1 = @(alpha) [R_B1(alpha), r_B1_inB; 0 0 0 1];

% Create the cumulative transformation matrix
% We will assume input of the configuration vector q = [alpha, beta,
% gamma]'
% Hint: H_B3 is a product of the matrices above
H_B3 = @(q) H_B1(q(1)) * H_12(q(2)) * H_23(q(3)); 

% find the foot point position vector
% Hint: This H_cut function just cuts out the first three rows of an H
% matrix to help recover a 3*1 vector. 
% Then, r_XY_inX = @(q) H_cut(H_XZ(q))*[r_ZY_inZ; 1];
H_cut = @(H) H(1:3,:);
r_BF_inB = @(q) H_cut(H_B3(q)) * [r_3F_in3; 1];


%% Q3

% Calculate the foot point Jacobian as a fn of configuration vector 
% q = [alpha; beta; gamma]'

    
% what generalized velocity dq do you have to apply in a configuration q = [0;60°;-120°]
% to lift the foot in vertical direction with v = [0;0;-1m/s];
dr = [0; 0; -1];
qval = pi/180*([0; 60; -120]);
J = J_BF_inB(qval,H_B1,H_12,H_23,r_3F_in3);
dq = J\dr;

fprintf('Q3: Target velocity r_dot = [%0.1f; %0.1f; %0.1f] m/s\n', dr(1), dr(2), dr(3));
fprintf('in current configuration q = [%0.1f; %0.1f; %0.1f] deg\n', qval(1)*180/pi, qval(2)*180/pi, qval(3)*180/pi);
fprintf('Requires qdot = [%0.1f; %0.1f; %0.1f] deg/s\n', dq(1)*180/pi, dq(2)*180/pi, dq(3)*180/pi);
fprintf('\n\n');


%% Q4

% write an algorithm for the inverse kinematics problem to
% find the generalized coordinates q that gives the end effector position 
% rGoal = [0.2,0.5,-2]' and store it in qGoal
q0 = pi/180*([0;-30;60]);
rGoal = [0.2;0.5;-2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% enter here your algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i_max = 200;
r_tolerance = 1e-6;

q = q0;
r = r_BF_inB(q);
r_error = rGoal-r;
i = 1;
while max(abs(r_error)) > r_tolerance
    q = q +  pinv(J_BF_inB(q,H_B1,H_12,H_23,r_3F_in3))*r_error;
    r = r_BF_inB(q);
    r_error = rGoal-r;
    i = i+1;
    if i >= i_max
        break;
    end
end
 
qGoal = q;
fprintf('Q4: Inverse kinematics for rGoal = [%0.1f; %0.1f; %0.1f]\n', rGoal(1), rGoal(2), rGoal(3));
fprintf('qGoal = [%0.1f; %0.1f; %0.1f] deg, found in %d iterations\n', qGoal(1)*180/pi, qGoal(2)*180/pi, qGoal(3)*180/pi, i-1);
     
 
%% Q5

% Write an algorithm for the inverse differential kinematics problem to
% find the generalized velocities dq to follow a circle in the body xz plane
% around the start point rCenter with a radius of r=0.5 and a 
% frequeny of 1Hz. The start configuration is q =  pi/180*([0,-60,120])'
q0 = pi/180*([0,-60,120])';
dq0 = zeros(3,1);
rCenter = r_BF_inB(q0);
radius = 0.5;
f = 0.25;
rGoal = @(t) rCenter + radius*[sin(2*pi*f*t),0,cos(2*pi*f*t)]';
drGoal = @(t) 2*pi*f*radius*[cos(2*pi*f*t),0,-sin(2*pi*f*t)]';

% define here the time resolution
deltaT = 0.01;
timeArr = 0:deltaT:1/f;
Kp = 10;

% q, r, and rGoal are stored for every point in time in the following arrays
qArr = zeros(3,length(timeArr));
rArr = zeros(3,length(timeArr));
rGoalArr = zeros(3,length(timeArr));

q = q0;
dq = dq0;
for i=1:length(timeArr)
    t = timeArr(i);
    % data logging, don't change this!
    q = q+deltaT*dq;
    qArr(:,i) = q;
    rArr(:,i) = r_BF_inB(q);
    rGoalArr(:,i) = rGoal(t);
    
    % controller: 
    % step 1: create a simple p controller to determine the desired foot
    % point velocity
    v = drGoal(t)+Kp*(rGoal(t)-r_BF_inB(q));
    % step 2: perform inverse differential kinematics to calculate the
    % gneralized velocities
    dq = J_BF_inB(q,H_B1,H_12,H_23,r_3F_in3)\v;
    
end

plotTrajectory(timeArr, qArr, rArr, rGoalArr, true);

function J = J_BF_inB(q,H_B1,H_12,H_23,r_3F_in3)
    
    HB1 = H_B1(q(1));
    H12 = H_12(q(2));
    H23 = H_23(q(3));
    H3F = [eye(3), r_3F_in3 ; 0 0 0 1];

    N = [[1,0,0]',[0,1,0]',[0,1,0]'];

    HB2 = HB1*H12;
    HB3 = HB2*H23;
    HBF = HB3*H3F;

    J = zeros(3,3);
    J(:,1)  = cross(HB1(1:3,1:3) * N(:,1), HBF(1:3,4)-HB1(1:3,4));
    J(:,2)  = cross(HB2(1:3,1:3) * N(:,2), HBF(1:3,4)-HB2(1:3,4));
    J(:,3)  = cross(HB3(1:3,1:3) * N(:,3), HBF(1:3,4)-HB3(1:3,4));
end

