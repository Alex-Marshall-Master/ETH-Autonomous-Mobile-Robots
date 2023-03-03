function plotTrajectory(timeArr, qArr, rArr, rGoalArr, videoFlag)
if ~exist('videoFlag','var')
    videoFlag = false;
end

r_B3_inB = @(alpha,beta,gamma)[...
    - sin(beta);...
  sin(alpha)*(cos(beta) + 1) + 1;...
  -cos(alpha)*(cos(beta) + 1)];


r_B2_inB = @(alpha,beta,gamma)[...
    sym(0);...
  sin(alpha)*(1) + 1;...
  -cos(alpha)*(1)];
        

rKArr = zeros(3,length(timeArr));
rHArr = zeros(3,length(timeArr));
for i=1:length(timeArr)
    q = qArr(:,i);
    rKArr(:,i) = r_B3_inB(q(1),q(2),q(3));
    rHArr(:,i) = r_B2_inB(q(1),q(2),q(3));
end

figure('position',[0,0,1000,400])
subplot(121)
hold on
axis equal
xlim([-0.7,1])
ylim([min(rGoalArr(3,:))-0.1,-0.50])
xlabel('x axis [m]')
ylabel('z axis [m]')
title('trajectory following')
pA = plot(rArr(1,:),rArr(3,:),'r');
pG = plot(rGoalArr(1,:),rGoalArr(3,:),'b');
pAA = plot(rArr(1,1),rArr(3,1),'ro');
pGG = plot(rGoalArr(1,1),rGoalArr(3,1),'bo');
pL1 = plot([rKArr(1,1),rArr(1,1)],[rKArr(3,1),rArr(3,1)],'k-.');
pL2 = plot([rHArr(1,1),rKArr(1,1)],[rHArr(3,1),rKArr(3,1)],'k-.');
pL3 = plot([0,rHArr(1,1)],[0,rHArr(3,1)],'k-.');
legend([pA, pG, pL1], {'actual position','target position','leg configuration'});


subplot(122)
xlabel('time [s]')
ylabel('position [m]')
title('trajectory following')
hold on
plot(timeArr,rArr(3,:),'r')
plot(timeArr,rGoalArr(3,:),'b')
legend('z-position [m]','z-target [m]')

if videoFlag
    for i = 1:length(timeArr)
        set(pAA, 'XData', rArr(1,i), 'YData', rArr(3,i));
        set(pGG, 'XData', rGoalArr(1,i), 'YData', rGoalArr(3,i));
        set(pL1, 'XData', [rKArr(1,i),rArr(1,i)], 'YData', [rKArr(3,i),rArr(3,i)]);
        set(pL2, 'XData', [rHArr(1,i),rKArr(1,i)], 'YData', [rHArr(3,i),rKArr(3,i)]);
        set(pL3, 'XData', [0, rHArr(1,i)], 'YData', [0, rHArr(3,i)]);
        pause(0.02);
    end
end