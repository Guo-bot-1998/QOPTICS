clear,clc

t = 0; j = 1;  
c =3e8;     % light speed
dx = 6e-7;  dt = 2e-15;           
L0 = 5e-4;
x = 0:dx:L0;
sigma = 3e-5/2.35482;  %FWHM = 2.35482*sigma = 30nm
y0 = exp(-(x-L0/2).^2/(2*sigma^2));  %Gaussian = a*exp(-(x-avg)^2/(2*sigma^2))                    
y = zeros(1,length(x));


hf_advec = figure('Name','Advection');    
hold on
hplot = plot(x,y,'LineWidth',2);
% plot(x,y+0.5,'black--')

xlabel('Position(m)');ylabel('Amplitude(m)');
i = 2:length(y);
axis([min(x) 3e-3 0 1])   %==== Inital axis

tic
while ishandle(hf_advec) && t <= 1e-11
    t = j*dt;
    
    if j> length(y0/2)
        % moving x-axis
    x(1) = []; x = [x x(length(x)) + dx];
        y(i-1) = y(i);y(length(y)) = 0;
    end
    y = advection1D(y,dx,dt,c,j,y0);
    j = j+1;
    if mod(j,1) == 0   
        set(hplot,'YData',y);
        set(hplot,'XData',x);
        titlestr = sprintf('time(s):%s',t);
        title(titlestr,'FontName','Arial','FontSize',10)
%        axis([min(x) max(x) 0 1])   %======= fixing axis on x line
        drawnow
    end
                 
end
toc

function [newY] = advection1D(oldY,dx,dt,c,j,y0)
% space backward difference 
    n = length(oldY);
    i = 2:n-1;
    if j<= length(y0)
        newY(1) = y0(j);
    else
        newY(1) = 0;
    end
    newY(n) = oldY(n) - (dt/dx)*c*(oldY(n)-oldY(n-1));
    newY(i) = oldY(i) - (dt/dx)*c*(oldY(i)-oldY(i-1));      
end




