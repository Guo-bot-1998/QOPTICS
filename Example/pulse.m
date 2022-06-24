clc,clear
t = 0; j = 0;
dx = 1e-15;  dt = 1e-15;           % c*dt/dx
c = 1;                         % wave speed
x = 0:dx:2e-10;
y = zeros(1,length(x));
value = 0;
Switch = 0;
n = 100;                         % number of pulses
for i = 1:length(x)
% Generate 100 pulses with width 100femto
 if ~mod(i*dx,1e-13)
     value = ~value;
     Switch = Switch + 1;
 end
 if Switch >= 2*n
     value = 0;
 end
 if value
     y(i) = 1;
 else 
     y(i) = 0;
 end
end

hf_advec = figure('Name','Advection');    
hold on
hplot = plot(x,y,'LineWidth',2);
xlabel('Position(m)');ylabel('Amplitude(m)');
axis([0 4e-13 0 1])   %%%%%%%%%%%%%%%%%



while ishandle(hf_advec)  && t <= 100*1e-15
    y = advection1D(y,dx,dt,c);
    if ishandle(hf_advec)
    set(hplot,'YData',y)
    j = j+1;
    t = j*dt
    end
     pause(0.0000000002)
end          


function [newY] = advection1D(oldY,dx,dt,c)
% backward difference 
% i starts from 2(matlab ��e��������0)
    n = length(oldY);
    i = 2:n-1;
    newY(i) = oldY(i) - (dt/dx)*c*(oldY(i)-oldY(i-1));      
    newY(n) = oldY(n) - (dt/dx)*c*(oldY(n)-oldY(n-1));
end