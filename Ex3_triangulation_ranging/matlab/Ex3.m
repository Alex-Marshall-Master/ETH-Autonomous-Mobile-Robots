% Set our laser ranger instrument fixed parameters
f = 0.5;
L = 1.0;

% Q1a Laser ranging measurement equation
range = @(x) f * L ./ x;

% Q1b Plot ranges as a function of distance
xx = linspace(0.05, 1.0, 50);
DD = range(xx);
plot(DD, xx, '.-');
xlabel('Range D');
ylabel('PSD measurement location x');
title('Laser ranging');

% Q2
sigma_x = 0.1;
range_uncertainty = @(x) f * L *sigma_x ./ (x.*x);

sDD = range_uncertainty(xx);
figure();
semilogy(DD, sDD, '.-');
xlabel('Range D');
ylabel('Measurement std \sigma_D');
title('Laser ranging with noise');
