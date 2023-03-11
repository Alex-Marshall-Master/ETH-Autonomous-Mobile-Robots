function plotOmnibot(F, phi_dot, dt)
% INPUT:    F - 3 x 3 forward kinematics matrix
%           phi_dot - 3 x n matrix of stacked wheel speeds
%           dt - time step in sec. Full simulation length is n*dt


% Figure setup
figure(1);
clf;
hold on;
grid on;
axis equal;
X = max(abs(xlim));
Y = max(abs(ylim));
xlim([-X,X]);
ylim([-Y,Y]);
xlabel('x (m)')
ylabel('y (m)')

% This loop performs simple Euler integration of
% the differential forward kinematics and plots 
% a coordinate frame of the robot each time step.
% The red line is the robot X axis and the green line
% is the robot Y axis. The robot starts at the origin
% with an orientation of zero.

% Initial robot state
xi = [0;0;0];

% Define some local positions for plotting the robot axes (these are in the
% robot frame)
p1 = [0.1;0];
p2 = [0;0.1];

steps = size(phi_dot, 2);

% Save an array of the full state history
fullState = zeros(3, steps+1);      
fullTime = 0:dt:steps*dt;
fullState(:,1) = xi;

% Rotation matrix for robot angle theta
R_RW = @(theta) [ cos(theta), sin(theta); -sin(theta), cos(theta) ];

Tp0 = xi(1:2);
R = R_RW(xi(3));
Tp1 = R'*p1 + xi(1:2);
Tp2 = R'*p2 + xi(1:2);
h_Rx = plot([Tp0(1),Tp1(1)], [Tp0(2),Tp1(2)],'r-','linewidth',2);
h_Ry = plot([Tp0(1),Tp2(1)], [Tp0(2),Tp2(2)],'g-','linewidth',2);
h_Rp = plot(Tp0(1), Tp0(2),'k-');

for i = 1:steps   
    % Propagate xi forward in time
    RR = [R, [0;0]; [0,0,1]];
    xi = xi + RR' * F * phi_dot(:,i) * dt;
    R = R_RW(xi(3));
        
    pause(dt);
    % plot the current location
    Tp0 = xi(1:2);
    Tp1 = R'*p1 + xi(1:2);
    Tp2 = R'*p2 + xi(1:2);
    set(h_Rx, 'XData', [Tp0(1),Tp1(1)], 'YData', [Tp0(2),Tp1(2)]);
    set(h_Ry, 'XData', [Tp0(1),Tp2(1)], 'YData', [Tp0(2),Tp2(2)]);
    set(h_Rp, 'XData', fullState(1,1:i), 'YData', fullState(2,1:i));
    
    fullState(:,i+1) = xi;
end
      
figure(2);
subplot(311)
plot(fullTime,fullState(1,:));
xlabel('time (s)');
ylabel('x (m)');
ylim([-X,X])
subplot(312)
plot(fullTime,fullState(2,:));
xlabel('time (s)');
ylabel('y (m)');
ylim([-Y,Y])
subplot(313)
plot(fullTime, fullState(3,:));
xlabel('time (s)');
ylabel('theta (radians)');
ylim([-pi,pi])
