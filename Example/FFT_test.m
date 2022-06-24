
% Making f-spectrum of sine 
% if signal is align at the tail and the head of next signal, the result is ideal(only one
% frequency component)
dt = 0.001; Nt = 20000; Lt = (Nt)*dt; t = dt:dt:Lt;               % time parameter      
df = 1/Lt; f = -1/(2*dt):df:1/(2*dt)-1*df;                        

S =  0*sin(2*pi*40*t) + 10*sin(2*pi*49*t)+0*cos(2*pi*235.74*t);         % input signal
X = S + 0*randn(size(t)); % while noise with variance 
% plot the input siganl
figure(1)
plot(t(1:800), X(1:800),'-*')
title('Signal X(t)')
xlabel('t (seconds)')
ylabel('X(t)')
%Fasr Fourier Transform
Y = fft(X);



%%
% Making f-spectrum of Gaussian 
dt = 0.001; Nt = 40000+1; Lt = (Nt-1)*dt; x = -Lt/2:dt:Lt/2-0*dt;               % time parameter      
dk = 1/Lt; k = -1/(2*dt):dk:1/(2*dt)-0*dk;

std = 1;
a = 1/(2*std^2);
S = exp(-a*(x.^2));

figure(3)
plot(x, S)
hold on
str = sprintf('Gaussian function,variance:%.3f',std^2);
xlim([-5 5])
title(str)
xlabel('x (m)')
ylabel('X(t)')
%Fasr Fourier Transform
Y = fft(S);
y = sqrt(pi/a)*exp(-(2*pi*k).^2/(4*a));
plot(k,y)
%%

P2 = abs(Y/Nt);   %normalized
P2 = 2*P2;

figure(2)
hold on
plot(f,fftshift(P2),'-*','MarkerSize',5) 
xlim([0 f(end)])
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(k)|')