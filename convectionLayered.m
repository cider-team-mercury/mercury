clear all

R=[3300; 3200;1500;0]*1e3; %Radius; m
% Tu=[1600,3000]; %temperature at the base of the thermal boundary layer (vector)
Ta0=[1500; 1200; 3000]; %initial average temperatures
Ts=300;

%% Geometry
A=4*pi*R.^2; %Area; m^2
Vc=4*pi/3.*R.^3; %cumulative volume; m^3;
V=diff(-Vc); %volume in each layer
n=numel(V); %number of layers

%%
% Material properties
temp=ones(size(V));
mu = [1;1;1]; %conversion factor in each layer (vector); STO table 13.2
Rac=2e3*temp; %critical Rayleigh number
eta=0.9e20*temp; %viscosity Pa.s
rho=[3;3;8]*1e3; % density; kg/m^3
H0=[1e-7;1e-7;0]./rho; %heat production W/m^3
lambda=[1.38e-17;1.38e-17;0];
nu=eta./rho; %kinematic viscosity; m^2/s
k=3*temp; %thermal conductivity; W/m/K
K=(1e-6)*temp; %thermal diffusivity; m^2/s
Cp=k./(rho.*K).*temp; %heat capacity; (J/(K-g))
a=(2e-5)*temp; %thermal expansion; 1/K
g=3*temp; %acceleration of gravity; m/s^2;
beta=(1/3)*temp; %parameter; Nu=Ra^(1/beta);
gamma=1./(1+beta); 

M=rho.*V; %mass in each layer (kg)
Myr=1e6*3e7; %convert s to Myr

% dependent variable:
%% Temperature-dependent viscosity, theta with temperature dependent viscosity

A0=7e4 ; % K, From TSO p 591
nu0=1.65e2 ; % m2/s , from TSO p 591. Kinematic viscosity.
visc=@(Ta)(nu0*exp(A0./Ta));
Theta=@(Ta)(k.*((a.*g./(K.*visc(Ta).*Rac)).^beta)); %temperature drop to heat flux conversion factor


%% Ta has time series of average temperature in each layer
Tuf=@(Ta)Ta./mu;
Tlf=@(Ta)2*Ta-Tuf(Ta);
H=@(t)H0.*exp(-lambda*t);

[t,Ta]=ode45(@(t,Ta)convectionODE(t,Ta,Tuf,Tlf,Ts,Theta,gamma,A,M,H,Cp,n), [0,4500*Myr],Ta0);

% % %% Postprocessing

figure
subplot(2,1,1)
plot(t,Ta(:,1),'b-')
hold on
plot(t,Ta(:,2),'r-')
hold on
plot(t,Ta(:,3),'g-')

for ii=1:length(t)
    hout(ii,:) = H(t(ii));
end
subplot(2,1,2)
plot(t,hout(:,2),'r')

figure
imagesc(Ta')
colorbar()
% % nt=size(Ta,1);
% % % more temperatures
% % Tu=Ta./repmat(mu',[nt,1]); %base of the upper BL
% % Tl=2*Ta-Tu; %top of the lower BL
% % % Boundary temperatures
% % Tb=NaN(nt,n+1);
% % Tb(:,1)=Ts; % surface temperature
% % %ThetaMat=repmat(Theta(Ta)',[nt,1])
% % ThetaMat=Theta(Ta);
% % for i=2:n;
% %     Tb(:,i)=(ThetaMat(:,i).*Tu(:,i)+ThetaMat(:,i-1).*Tl(:,i-1))./(ThetaMat(:,i)+ThetaMat(:,i-1));
% % end
% % Tb(:,n+1)=Tl(:,end); %made-up temperature at the center of the Earth;
% % 
% % %Temperature drop (K)
% % DTu=Tu-Tb(:,1:end-1);
% % DTl=Tb(:,2:end)-Tl;
% % % Heat flux (W/m^2)
% % Qu=sign(DTu).*ThetaMat.*(abs(DTu)).^(4/3);
% % Ql=sign(DTl).*ThetaMat.*(abs(DTl)).^(4/3);
% % % boundary Layer thickness
% % kMat=repmat(k',[nt,1]);
% % du=kMat.*DTu./Qu;
% % dl=kMat.*DTl./Ql;
% % 
% % Ru=repmat([R(1:end-1)]',[nt,1])-du; %radius at the bottom of the upper BL
% % Rl=repmat([R(2:end)]',[nt,1])+dl; %radius at the top of the lower BL
% % 
% % HMat=repmat(H0',[nt,1]).*exp(-repmat(lambda',[nt,1]).*repmat(t,[1,n]));
% % VMat=repmat(V',[nt,1]);
% % rhoMat=repmat(rho',[nt,1]);
% % Ht=sum(HMat.*VMat.*rhoMat,2)
% % 
% % %%
% % Tmat=NaN(nt,3*n-0);
% % Rmat=NaN(nt,3*n-0);
% % for i=1:n;
% %     Tmat(:,i*3-2)=Tb(:,i);
% %     Tmat(:,i*3-1)=Tu(:,i);
% %     Tmat(:,i*3-0)=Tl(:,i);
% %     Rmat(:,i*3-2)=repmat(R(i),[nt,1]);
% %     Rmat(:,i*3-1)=Ru(:,i);
% %     Rmat(:,i*3-0)=Rl(:,i);
% % end
% % %
% % figure(1); clf
% % subplot 221;
% % plot(t/Myr,Qu'*1e3)
% % xlabel('Wime (Myr)');
% % ylabel('Heat flux (mW/m^2)');
% % title('Top of each layer');
% % legend(num2str([1:n]'))
% % hold on;
% % plot(t/Myr,Ht./A(1)*1e3,'k','linewidth',2);
% % 
% % subplot 222;
% % plot(t/Myr,Ta')
% % xlabel('time (Myr)');
% % ylabel('temperature (K)');
% % title('Average for each layer');
% % legend(num2str([1:n]'))
% % 
% % subplot 223;
% % plot(Tmat',Rmat'/1000)
% % xlabel('Temperature (K)');
% % ylabel('Radius (km)');
% % 
% % subplot 224;
% plot(t/Myr,Tmat')
% xlabel('time (Myr)');
% ylabel('temperature (K)');
% 
% 
% 
% 





    