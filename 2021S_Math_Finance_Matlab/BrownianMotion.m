%Matlab c style code for Brownian Motion
%matlab has an imbedded functions for Brownian Motion.
%It is also possible to
%neatly write in Matlab a 4 liner:

% N = 1000;
% dW = randn(1,N);
% x = cumsum(dW);
% plot(x);

%However we resort to a slow
%c style code with loops for educational purposes

M=40000; %number of trajectories of Brownian motion

N=250;%Number of steps in one trajectory
X0=50; %initial point
T=1;  %Final Time in years in trajectory

mu=0.01;
sigma=0.22;

dt=T/N; %time step
Sqrtdt=sqrt(dt);

%X(j,:) j-th trajectory of Brownian Motion
X(1:M,1)=X0; % Initial value of Brownian Motion  X(j,1)=X0 for all j=1:M
             %here index starts with 1 and not 0 as in Matlab array index should be
             %positive
Y50(1:M)=0;
Y51(1:M)=0;
for j=1:M  %generate M trajectories
    for i = 2:N+1  %generate j-th trajectory
        X(j,i)=X(j,i-1)+mu*X(j,i-1)*dt+Sqrtdt*randn*power(X(j,i-1),0.8)*sigma;
    end
    
    if X(j,N+1)> 50
        Y50(j)= (X(j,N+1)-50)*exp(-1*mu*T);
    end

    if X(j,N+1)> 51
        Y51(j)= (X(j,N+1)-51)*exp(-1*mu*T);
    end
end

t=0:dt:T;
plot(t,X(:,:));

price_50 = mean(Y50(1:M));
price_51 = mean(Y51(1:M));
