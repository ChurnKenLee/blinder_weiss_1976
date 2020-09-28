clear; clc;

% cd '/Users/adriencouturier/Dropbox/Adrien-Ben_RA/Human Capital'

%% Parameters
sigma = 2; %CRRA utility with parameter s
theta = .2; %learning ability parameter
r = 0.05; %interest rate
rho = 0.05; %discount rate
delta = 0.07; %depreciation rate for human capital
w = 1; %wage rate
alpha = 0.5; %elasticity of human capital production function


crit = 1e-6;
max_itr = 200;
dt = 10; %time step

options = optimset('TolFun', 1e-15, 'MaxIter', 400, 'Display', 'off'); %fsolve options

%% Functions
if (sigma==1)
    u =@(c) log(c);
    u_prime = @(c) 1./c;
    u_prime_inv = @(x) 1./x;
else
    u =@(c) c.^(1-sigma)/(1-sigma);
    u_prime = @(c) c.^(-sigma);
    u_prime_inv = @(x) x.^(-1/sigma);
end
 
%% Steady state

sstar = (alpha*delta)/(rho+delta);
hstar = (sstar^alpha*theta/delta)^(1/(1-alpha));

%% Constructing the Grids
grid_curv = 1; %1 for linear grid
na = 100; %number of grid points in the a dimension
nh = 40; %number of grid points in the h dimension

hmin = hstar*0.7; hmax=hstar*1.3; %if hmin too small it breaks at a=0
h1=hmin+(hmax-hmin)*linspace(0,1,nh).^(1/grid_curv);
dh = (hmax-hmin)/(nh-1);
h = repmat(h1,na,1);

amin = 0; amax = 5;
a1=amin+(amax-amin)*linspace(0,1,na)'.^(1/grid_curv);
da = (amax-amin)/(na-1);
a= repmat(a1,1,nh);

%% Initialize Policy Functions and Fwd/Bwd Derivatives
dVaF = zeros(na,nh);
dVaB = zeros(na,nh);
dVhF = zeros(na,nh);
dVhB = zeros(na,nh);


m = zeros(na, nh); %indicator for whether we use a st st update in iteration
Id = zeros(na, nh); %indexes of upwind updates

dist = zeros(max_itr,1); %distance function to track convergence

%% Initial guess: staying put
S0 = subplus(min((delta/theta)^(1/alpha).*(h.^(1/alpha-1)),1));
C0 = r*a+w*h.*(1-S0);
dVa0 = u_prime(C0);
dVh0 = (w/(theta*alpha))*((h.*S0).^(1-alpha)).*dVa0;
V0 = u(C0)/rho;
V=V0;
C=C0;

%% Transition dynamics parameters

color2 = [0.8500, 0.3250, 0.0980];
LineWidth = 3;

T = 300;
dtime = .5;
t = linspace(0,dtime*T,T)';
a_path = zeros(T,1);
h_path = zeros(T,1);
c_path = zeros(T,1);
s_path = zeros(T,1);

smoothing_method = 'cubic';

xia = .8;
xih = 0;
a_path(1,1) = amax*xia+amin*(1-xia);
h_path(1,1) = hmax*xih+hmin*(1-xih);

%% Options
Display_transition  = 0;
Display_policy  = 1;

%% Iteration
tic;

for i=1:max_itr
    v=V;

    % forward difference
    dVaF(1:na-1,:) = (v(2:na,:)-v(1:na-1,:))/da;
    dVaF(na,:)=dVaF(na-1,:);

    dVhF(:,1:nh-1) = (v(:,2:nh)-v(:,1:nh-1))/dh;
    dVhF(:,nh) = dVhF(:,nh-1);
    

    % backward difference
    dVhB(:,2:nh) = (v(:,2:nh)-v(:,1:nh-1))/dh;
    dVhB(:,1) = dVhB(:,2);

    dVaB(2:na,:) = (v(2:na,:)-v(1:na-1,:))/da;
    dVaB(1,:) = dVaB(2,:);
    
    
    % check that the value function estimation remains increasing in wealth
    check1 = dVaF <0; check2 = dVaB <0;
    if (sum(sum(check1))>0)
        fprintf('No convergence. Parameters possibly out of range. \n');
        break;
    end

    
    %consumption with forward and backward difference (only depends on Va)
    cF = u_prime_inv(dVaF);
    cB = u_prime_inv(dVaB);

    sBB = min(subplus((theta*alpha/w)*dVhB.*(h.^(alpha-1))./dVaB),1).^(1/(1-alpha));
    sBF = min(subplus((theta*alpha/w)*dVhF.*(h.^(alpha-1))./dVaB),1).^(1/(1-alpha));
    sFB = min(subplus((theta*alpha/w)*dVhB.*(h.^(alpha-1))./dVaF),1).^(1/(1-alpha));
    sFF = min(subplus((theta*alpha/w)*dVhF.*(h.^(alpha-1))./dVaF),1).^(1/(1-alpha));

    %drift matrices
    a_muBB = r*a + w*h.*(1-sBB)-cB;
    a_muBF = r*a + w*h.*(1-sBF)-cB;
    a_muFB = r*a + w*h.*(1-sFB)-cF;
    a_muFF = r*a + w*h.*(1-sFF)-cF;

    h_muBB = theta*(sBB.*h).^(alpha)-delta*h;
    h_muBF = theta*(sBF.*h).^(alpha)-delta*h;
    h_muFB = theta*(sFB.*h).^(alpha)-delta*h;
    h_muFF = theta*(sFF.*h).^(alpha)-delta*h;

   % fwd/bwd consistency indicator matrices
   I_BB = a_muBB <0 & h_muBB <0;
   I_BB(1,:) = 0;
   I_BB(:,1) = 0;

   I_BF= a_muBF <0 & h_muBF >0;
   I_BF(1,:) =0;
   I_BF(:,nh) = 0;

   I_FB= a_muFB >0 & h_muFB <0;
   I_FB(na,:) = 0;
   I_FB(:,1) = 0;

   I_FF= a_muFF >0 & h_muFF >0;
   I_FF(na,:)=0;
   I_FF(:,nh)=0;


   for j=1:nh
       for k=1:na
           [m(k,j),Id(k,j)]= max([I_FF(k,j), I_FB(k,j), I_BF(k,j), I_BB(k,j)]);           
       end
   end


   %imposing boundary conditions (from the a_min state constraint) on the
   %upper edge, i.e. a_min
   for j=1:nh
       if (m(1,j)==0)
           dVhj = dVhF(1,j);
           fn=@(x) r*a1(1)+w*h1(j)*(1-((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./x).^(1/(1-alpha)))-x^(-1/sigma);
           dVamaxj = 2*(theta*dVhj/(w*hmin))^3;

           try
               dVaj = fzero(fn, [1e-10,dVamaxj]);
           catch
               dVaj = fsolve(fn, dVaF(1,j), options);
           end

           sj = min(subplus((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./dVaj),1).^(1/(1-alpha));
           mu_hj = theta*(sj*h1(j))^alpha-delta*h1(j);

           if ( mu_hj>= 0 && sj >0 && sj <1 && j~=nh)
              m(1,j)=1;
              Id(1,j)=3;
              dVaB(1,j) = dVaj;
              h_muBF(1,j) = mu_hj;
              a_muBF(1,j) = r*a1(1)+w*h1(j)*(1-sj)-u_prime_inv(dVaj);
           else
               dVhj = dVhB(1,j);
               fn=@(x) r*a1(na)+w*h1(j)*(1-((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./x).^(1/(1-alpha)))-x^(-1/sigma);
               dVaj = fsolve(fn, dVaF(1,j), options);

               sj = min(subplus((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./dVaj),1).^(1/(1-alpha));
               mu_hj = theta*(sj*h1(j))^alpha-delta*h1(j);

               if (mu_hj <= 0 && sj>0 && sj<1 && j~=1)
                   m(1,j)=1;
                   Id(1,j)=4;
                   dVaB(1,j) = dVaj;
                   h_muBB(1,j) = mu_hj;
                   a_muBB(1,j) = r*a1(1)+w*h1(j)*(1-sj)-u_prime_inv(dVaj);
               end
           end               
       end
   end


   %imposing boundary conditions on the lower edge, i.e. a_max
   for j=1:nh
       if (m(na,j)==0)
           dVhj = dVhF(na,j);
           fn=@(x) r*a1(na)+w*h1(j)*(1-((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./x).^(1/(1-alpha)))-x^(-1/sigma);

           try
               dVaj = fzero(fn, [1e-10,dVamaxj]);
           catch
               dVaj = fsolve(fn, dVaF(1,j), options);
           end

           sj = min(subplus((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./dVaj),1).^(1/(1-alpha));
           mu_hj = theta*(sj*h1(j))^alpha-delta*h1(j);

           if (mu_hj >= 0 && sj>0 && sj<1 && j~=nh)
              m(na,j)=1;
              Id(na,j)=1;
              dVaF(na,j) = dVaj;
              h_muFF(na,j) = mu_hj;
              a_muFF(na,j) = r*a1(na)+w*h1(j)*(1-sj)-u_prime_inv(dVaj);
           else

               dVhj = dVhB(na,j);
               fn=@(x) r*a1(na)+w*h1(j)*(1-((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./x).^(1/(1-alpha)))-x^(-1/sigma);
               dVaj = fsolve(fn, dVaF(1,j), options);

               sj = min(subplus((theta*alpha/w)*dVhj.*(h1(j).^(alpha-1))./dVaj),1).^(1/(1-alpha));
               mu_hj = theta*(sj*h1(j))^alpha-delta*h1(j);

               if (mu_hj <= 0 && sj>0 && sj<1 && j~=1)
                   m(na,j)= 1;
                   Id(na,j)= 2;
                   dVaF(na,j) = dVaj;
                   h_muFB(na,j) = mu_hj;
                   a_muFB(na,j) = r*a1(na)+w*h1(j)*(1-sj)-u_prime_inv(dVaj);
               end
           end               
       end
   end


   
   %imposing boundary conditions on the left edge, i.e. h_min
    for j=1:na
       if (m(j,1)==0)
           sj = S0(j,1);
           dVaj = dVaF(j,1);
           cj = u_prime_inv(dVaj);
           aj_mu = r*a1(j)+w*h1(1)*(1-sj)-cj;

            if (aj_mu >= 0 && j~=na)
              dVhj = (w/(alpha*theta))*(sj*h1(1))^(1-alpha)*dVaj;
              m(j,1)=1;
              Id(j,1)=2;
              dVhB(j,1) = dVhj;
              h_muFB(j,1) = -delta*h1(1)+theta*(sj*h1(1))^alpha;
              a_muFB(j,1) = aj_mu;
           else
               dVaj = dVaB(j,1);
               cj = u_prime_inv(dVaj);
               aj_mu = r*a1(j)+w*h1(1)*(1-sj)-cj;

               if (aj_mu <= 0 && j~=1)
                   dVhj = (w/(alpha*theta))*(sj*h1(1))^(1-alpha)*dVaj;
                   m(j,1)=1;
                   Id(j,1)=4;
                   dVhB(j,1) = dVhj;
                   h_muBB(j,1) = -delta*h1(1)+theta*(sj*h1(1))^alpha;
                   a_muBB(j,1) = aj_mu;
               end
           end               
       end
   end
 
   
   %imposing boundary conditions on the right edge, i.e. h_max
   for j=1:na
       if (m(j,nh)==0)
           sj = S0(j,nh);
           dVaj = dVaF(j,nh);
           cj = u_prime_inv(dVaj);
           aj_mu = r*a1(j)+w*h1(nh)*(1-sj)-cj;
 
           if (aj_mu >= 0 && j~=nh)
              dVhj = (w/(alpha*theta))*(sj*h1(nh))^(1-alpha)*dVaj;
              m(j,nh)=1;
              Id(j,nh)=1;
              dVhF(j,nh) = dVhj;
              h_muFF(j,nh) = -delta*h1(nh)+theta*(sj*h1(nh))^alpha;
              a_muFF(j,nh) = aj_mu;
              
           else
               dVaj = dVaB(j,nh);
               cj = u_prime_inv(dVaj);
               aj_mu = r*a1(j)+w*h1(nh)*(1-sj)-cj;
 
               if (aj_mu <= 0 && j~=1)
                   dVhj = (w/(alpha*theta))*(sj*h1(nh))^(1-alpha)*dVaj;
                   m(j,nh)=1;
                   Id(j,nh)=3;
                   dVhF(j,nh) = dVhj;
                   h_muBF(j,nh) = -delta*h1(nh)+theta*(sj*h1(nh))^alpha;
                   a_muBF(j,nh) = aj_mu;
               end
           end
       end
   end


   dVa = (dVaF.*(Id <3) + dVaB.*(Id>2)).*m + dVa0.*(1-m);
   dVh = (dVhF.*(mod(Id,2)) + dVhB.*(1-mod(Id,2))).*m + dVh0.*(1-m);

   C = dVa.^(-1/sigma);

   mu_a = (a_muFF.*(Id==1)+a_muFB.*(Id==2)+a_muBF.*(Id==3)+a_muBB.*(Id==4)).*m;
   mu_h = (h_muFF.*(Id==1)+h_muFB.*(Id==2)+h_muBF.*(Id==3)+h_muBB.*(Id==4)).*m;


   %Building implicit update matrices
   diag = reshape(abs(mu_a)./da + abs(mu_h)./dh, [na*nh,1]); %main diagol
   
   diag_m1 = -reshape(subplus(-mu_a)./da, [na*nh,1]); %diagol -1
   diag_m1 = circshift(diag_m1, -1);

   diag_mn = -reshape(subplus(-mu_h)./dh, [na*nh,1]); %diagol -na
   diag_mn= circshift(diag_mn, -na);

   diag_p1 = -reshape(subplus(mu_a)./da, [na*nh,1]); %diagol +1
   diag_p1=circshift(diag_p1, 1);

   diag_pn = -reshape(subplus(mu_h)./dh, [na*nh,1]); %diagol +na
   diag_pn = circshift(diag_pn, na);
       
   A= spdiags([diag_mn diag_m1 diag diag_p1 diag_pn], [-na,-1,0,1,na], na*nh,na*nh);

   B= (1/dt+rho)*speye(na*nh,na*nh) + A;

   R = reshape(v./dt+u(C), [na*nh, 1]);
   V = reshape(B\R,[na,nh]);

   if max(abs(sum(A,2)))>10^(-9)
    disp('Improper Transition Matrix')
    sum(A,2)
    break
   end

   dist(i) = max(max(abs(V-v)));
   fprintf('Value function iteration %i, made distance %d.\n',i, dist(i));
   
   if (i > 1 && dist(i)>dist(i-1))
        disp('Value function is not converging')
        break
   end
   
   if dist(i)<crit
        disp('Value Function Converged, Iteration = ')
        disp(i)
        break
   end
     
end
S = min(subplus((theta*alpha/w)*dVh.*(h.^(alpha-1))./dVa),1).^(1/(1-alpha));
toc;


%% Transition dynamics

if Display_transition == 1

    for n=1:T-1
        % interpolate policy functions
        mu_a_policy = interp2(h1,a1,mu_a,h_path(n),a_path(n),smoothing_method);
        mu_h_policy = interp2(h1,a1,mu_h,h_path(n),a_path(n),smoothing_method);
        c_policy = interp2(h1,a1,C,h_path(n),a_path(n),smoothing_method);
        s_policy = interp2(h1,a1,S,h_path(n),a_path(n),smoothing_method);

        % assign values to time paths
        a_path(n+1) = a_path(n) + dtime*mu_a_policy;
        h_path(n+1) = h_path(n) + dtime*mu_h_policy;
        c_path(n) = c_policy;
        s_path(n) = s_policy;
    end

        c_policy = interp2(h1,a1,C,h_path(T),a_path(T),smoothing_method);
        s_policy = interp2(h1,a1,S,h_path(T),a_path(T),smoothing_method);
        c_path(T) = c_policy;
        s_path(T) = s_policy;

    figure(1) = figure('units','centimeters','outerposition',[10 10 16*1.2 9*1.2]);
    subplot(2,2,1)
    plot(t,a_path,'LineWidth',LineWidth);
    hold on
    plot(t,ones(T,1)*a_path(end))
    ylim([amin,amax]);
    title('Convergence path for financial capital')
    ylabel('Financial Capital','FontSize', 8)
    xlabel('Time','FontSize', 8)

    subplot(2,2,2)
    plot(t,h_path,'LineWidth',LineWidth);
    hold on
    plot(t,ones(T,1)*hstar)
    ylim([hmin,hmax]);
    title('Convergence path human capital')
    ylabel('Human Capital','FontSize', 8)
    xlabel('Time','FontSize', 8)

    subplot(2,2,3)
    plot(t,c_path,'Color',color2,'LineWidth',LineWidth);
    hold on
    plot(t,ones(T,1)*(r*a_path(end)+w*(1-sstar)*hstar));
    title('Convergence path for consumption')
    ylabel('Consumption','FontSize', 8)
    xlabel('Time','FontSize', 8)

    subplot(2,2,4)
    plot(t,s_path,'Color',color2,'LineWidth',LineWidth);
    hold on
    plot(t,ones(T,1)*sstar)
    ylim([0,1]);
    title('Convergence path time spent in education')
    ylabel('Education','FontSize', 8)
    xlabel('Time','FontSize', 8)

    set(gcf, 'Color', 'w');
    %export_fig 'HC_deterministic_transition.pdf'

end

%% Plots
if Display_policy == 1

    % export
    figure(2) = figure('units','centimeters','outerposition',[10 10 16*1.2 9*1.2]);

    subplot(1,2,1);
    surf(h1,a1,C)
    %shading interp    % interpolate colors across lines and faces
    title('Policy plot:  Consumption c')
    xlim([hmin,hmax]);
    ylim([amin,amax]);    
    ylabel('physical capital a','FontSize', 8)
    xlabel('human capital h','FontSize', 8)
    zlabel('policy c','FontSize', 8)
    
    subplot(1,2,2);
    surf(h1,a1,S)
    %shading interp    % interpolate colors across lines and faces
    title('Policy plot: fraction of time for learning s')
    xlim([hmin,hmax]);
    ylim([amin,amax]);
    ylabel('physical capital a','FontSize', 8)
    xlabel('human capital h','FontSize', 8)
    zlabel('policy s','FontSize', 8)

    
    set(gcf, 'Color', 'w');
    %export_fig 'HC_deterministic_policy.pdf'
    
end