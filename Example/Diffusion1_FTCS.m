dx = 0.1;
dt = 0.0005;
t = 0;
k = 10; L = 10;
x = 0:dx:L;
terms = 10;
yzero = zeros(1,length(x));
y = yzero +1;
y(1) = 0; y(length(x)) = 0;

hf_diffusion = figure('Name','DIffustion');
yyaxis left
hold on
hplotSimu = plot(x,yzero,'LineWidth',3);
hplotSol = plot(x,yzero,'r--','LineWidth',2);
% axis([-inf,inf,0,1.1]);
xlabel('Postion(cm)'); ylabel ('Heat(T)');

yyaxis right
hplotErr = plot(x,yzero,'black-.','LineWidth',0.5);
plot(x,yzero,'black-','LineWidth',0.5);
legend Numerical Analytical Error
ylabel('Difference between Solutions(Numer-Ant)');



isflim = 0;
%==================== Initialize video
loops = 500;
F(loops) = struct('cdata',[],'colormap',[]);  % Preallocate frame structre array
myVideo = VideoWriter('diffustion2224.avi'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)

j = 1;
while ishandle(hf_diffusion) && t < 0.1
    t = j*dt;
    y = diffusion1D(y,dx,dt,k);
    yh = DFsol_tn(x,terms,1,k,L);   
    sol_y = yh(t);    
    err = (sol_y - y);
    set(hplotSimu,'YData',y)
    set(hplotSol,'YData',sol_y)
    set(hplotErr,'YData',err)
    titlestr = sprintf('time:%.4f, k:%.1f, SOL terms: %d',t,k,terms);
    title(titlestr,'FontName','Arial','FontSize',10)
    drawnow
    pause(0.000001)
    % Animation part
    if isflim
        ax = gca;
        ax.Units = 'pixels';
        pos = ax.Position;
        ti = ax.TightInset;
        rect = [-ti(1), -ti(2), pos(3)+ti(1)+ti(3), pos(4)+ti(2)+ti(4)];
        try
        F(j) = getframe(gcf);
        writeVideo(myVideo, F(j));
        catch
        fprintf('miss%d\n',j)
        end
    end
    j = j+1;
end
close(myVideo)

function newY = diffusion1D(oldY,dx,dt,k)
    n = length(oldY);       
    i = 2:n-1;  
    newY(i) = oldY(i) + (dt/dx^2)*k*(oldY(i+1)-2*oldY(i)+oldY(i-1));
    %boundary
    newY(1) = 0; 
    newY(n) = 0; 
end


        