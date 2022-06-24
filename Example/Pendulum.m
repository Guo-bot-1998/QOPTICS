clear
%problems
%when the initial angle is 179degree, the pendulum will not go back.
%=Define the parameters
global g l
g = 9.8;
l = 9.8;
del_t = 0.001;
%=Initial conditions
initAngle = 10;  % degrees
theta0 = deg2rad(initAngle);        
w0 = 0;          % acceleration of angle
Y0 = [theta0 w0];           % vector
%=Create Plotting 
hf = figure('Name','Pendulum','Position',[1000 317.8 560 420]);
hold on
hpeuler = plot(Y0(1),Y0(2),'b*','MarkerSize',3);
hprk4 = plot(Y0(1),Y0(2),'ro','MarkerSize',3);
title('Simple pendulum');
xlabel('Theta (deg)');ylabel('Omega (deg/s)');
% hferr = figure('Name','Error','Position',[300 317.8 560 420]);
% hperr = plot(0,0);
% err_ = 0;t = 0;


Yold = Y0; Yold2 = Y0;

n = 10;                      % display at each "n" frames
i = 0; j = 0;

tic
while ishandle(hf)
    figure(hf);
    Ynow = Euler(del_t,Yold);
    Ynow2 = RK4(del_t,Yold); 
    i = i+1;
    Error = Ynow - Ynow2;    %Error
%    t(i) = i*del_t;
    if mod(i,n) == 0  
        plot(rad2deg(Ynow(1)),rad2deg(Ynow(2)),'b*','MarkerSize',3);
        plot(rad2deg(Ynow2(1)),rad2deg(Ynow2(2)),'ro','MarkerSize',3);
%        fprintf('%f sec\n',i*del_t);
%        figure(hferr);
%        err = abs(Ynow(1))+abs(Ynow(2)) - (abs(Ynow2(1))+abs(Ynow2(2)));
%        j = j+1;
%        err_(j) = err;
    end
    Yold = Ynow; Yold2 = Ynow2;
%     set(hperr,'Xdata',t);
%     set(hperr,'Ydata',err_);
    pause(0.0001)
end
toc
function ydot = F(y)
% The vector. [theta x] == [theta omega]
global g l
theta = y(1); w = y(2); 
 ydot = [w, -(g/l)*sin(theta)];

end

function next = Euler(h,now)
% Euler Method
% h = step
next = now + h.*F(now);
end
function next = RK4(h,now)
% RK4 Method
K1 = h*F(now);
K2 = h*F([now(1)+0.5*h, now(2)+0.5*K1]);
K3 = h*F([now(1)+0.5*h, now(2)+0.5*K2]);
K4 = h*F([now(1)+h, now(2)+K3]);
next = now + 1/6.*(K1+2*K2+2*K3+K4);

end
