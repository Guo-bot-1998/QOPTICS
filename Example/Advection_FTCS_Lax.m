clc,clear
dx = 0.1; dt = 0.01;      % c*dt/dx
c = 10;  k = 2*pi;         
x = 0:dx:10;    

y0 = sin(k*x);          
y = zeros(1,length(x)); yLax = y;
t = 0;

%dimensionless variable version
%pure numeric concern
dxx = 0.1; dtt = 0.1; 
xx = 0:dxx:62.8;
yy0 = sin((xx)); 
yy = zeros(1,length(xx)); tt = 0;

xx = xx/k;   % 變回有單位

hf_advec = figure('Name','Advection');
hold on
% hplotLax = plot(x,yLax,'r-','LineWidth',2);  
hplot = plot(x,y,'b*','LineWidth',0.5);
axis([-inf inf -1 1])                   %Axis
xlabel('Position(m)');ylabel('Amplitude(m)');
plot(x,y,'black--','LineWidth',1)      %初始y值 (x軸)
scatter(5,0,'black.','SizeData',300)   %零點
% legend('LaxMethod','FTBS')
hold off

figure(2)
hplot2 = plot(xx,yy,'b*','LineWidth',0.5);
axis([-inf inf -1 1])

j = 0;

while ishandle(hf_advec)  && tt < 1
    j = j+1; t = (j-1)*dt; tt = ((j-1)*dtt)/(c*k)        % 第一步初始狀態 t=(j-1)*dt = 0
    [y,yLax] = advection1D(y,yLax,dx,dt,c,j,y0);
    yy = dimensionless(yy,dxx,dtt,j,yy0);
    if ishandle(hf_advec)
    set(hplot,'YData',y)
    set(hplot2,'YData',yy)
    titlestr = sprintf('λ:%.1f, c:%.1f\ntime:%.4f',2*pi/k,c,t);
    title(titlestr,'FontName','Arial','FontSize',10)
%     set(hplotLax,'YData',yLax)

    drawnow
%     pause(0.1)
    end
end          
% close(myVideo)

function [newYY] = dimensionless(oldYY,dxx,dtt,j,yy0)
% To verify whether the dimensionless variable works.
    n = length(oldYY);
    i = 2:n-1;
%     if j <= length(yy0)
%         newYY(1) = yy0(j);
%     else
%         newYY(1) = 0;
%     end
    newYY(1) = sin((j-1)*dxx);
    newYY(i) = oldYY(i) - (dtt/dxx)*(oldYY(i)-oldYY(i-1));
    newYY(n) = oldYY(n) - (dtt/dxx)*(oldYY(n)-oldYY(n-1));
end

function [newY,newYLax] = advection1D(oldY,oldYLax,dx,dt,c,j,y0)
% ==== FTBS======
    n = length(oldY);
    i = 2:n-1;
    if j<= length(y0)
        newY(1) = y0(j);
    else
        newY(1) = 0;
    end
    newY(i) = oldY(i) - (dt/dx)*c*(oldY(i)-oldY(i-1));      
    newY(n) = oldY(n) - (dt/dx)*c*(oldY(n)-oldY(n-1));
    newYLax = methodLax;
    
    function newYLax = methodLax
    %===LaxMethod===
        newYLax(n) = oldYLax(n) - (dt/dx)*c*(oldYLax(n)-oldYLax(n-1));
        newYLax(i) = (oldYLax(i+1)+oldYLax(i-1))./2 -(c*dt./(2*dx)).*(oldYLax(i+1)-oldYLax(i-1));  
            if j<= length(y0)
                newYLax(1) = y0(j);
            else
                newYLax(1) = 0;
            end  
    end
end

