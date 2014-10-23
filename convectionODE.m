function dTadt=convectionODE(t,Ta,Tuf,Tlf,Ts,Theta,gamma,A,M,H,C,n);
%%
% th=Theta.^gamma
% Ta=@(T,mu)mu.*T; %average temperature in each layer (vector)
% Tl=@(Tu)Tu; %temperature at the top of the bottom boundary layer (vector)

%intial guess for boundary layer temps

% Ta=mu.*Tu;
Tu=Tuf(Ta);
Tl=Tlf(Ta);
% Determine boundary temperatures
Tb=Tl.*NaN;
Tb(1)=Ts; % surface temperature

% estimate the boundary layer temps
Tbu=Tb.*NaN;
Tbu(1) = (Tb(1)+Tu(1))/2;
Tbl=Tbu.*NaN;
Tbl(end)= Tl(end);
Tb(n+1)=Tl(end); %made-up temperature at the center of the Earth;
display('setting Tbu, Tbl')
for ii =2:n;
   Tbu(ii) = 0.75*Tu(ii)+0.25*Tl(ii-1); 
   Tbl(ii-1) = 0.25*Tu(ii)+0.75*Tl(ii-1);
end

% iterate until Tb's converge

for ii=1:3
    display('ii is '+ii)
    thetau=Theta(Tbu).^gamma;
    thetal=Theta(Tbl).^gamma;
    
    old=Tb;
    
    for jj = 2:n
        Tb(jj) = (thetal(jj-1)*Tl(jj-1) +thetau(jj)*Tu(jj))/(thetau(jj)+thetal(jj-1));
    end

    Tbu = (Tb(1:end-1)+Tu)/2;
    Tbl = (Tb(2:end)+Tl)/2;
    
    if (((sum((Tb-old).^2))^0.5)/mean(Tb)<0.1)
        thetau=Theta(Tbu).^gamma;
        thetal=Theta(Tbl).^gamma;
        display('Tbs converged')
        break
    end
end

dTu=Tu-Tb(1:end-1);
dTl=Tb(2:end)-Tl;


    
qt=sign(dTu).*thetau.*(abs(dTu)).^(4/3);
qb=sign(dTl).*thetal.*(abs(dTl)).^(4/3);

% Calculate new C for core



dTadt=((A(2:end).*qb... heat flux from next layer below
    -A(1:end-1).*qt... heat flux toward next layer above
    )./M... spread over the volume of each layer
    +H(t)... adding heat production W/kg; assumed constant layer
    )./C;... convert to temperature change
% dTadt=(H+(...
%     (A(2:end).*qb... Heat flux from next layer below
%     -A(1:end-1).*qt) ... Heat flux toward next layer above
%     ./M)./C);