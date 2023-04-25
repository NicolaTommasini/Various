clear;
%IMPORT DATA
global  Rate AssetPrice Maturity DividendYield OptSpec strike z real settle maturity

data = readtable('real_price_fb.xlsx');
M1 = data{:,1};
M2 = data{:,2};
M3 = data{:,3};
M4 = data{:,4};
M5 = data{:,5};
strike =(265:5:295);
real = transpose(data{:,:});
% figure(20)
% hold on;
% plot(strike, M1);
% plot(strike, M2);
% plot(strike, M3);
% plot(strike, M4);
% plot(strike, M5);
% title(' REAL CALL PRICE  ');
% hold off
%IMPORT PARAMETER
AssetPrice = 277;
Rate = 0.0;
DividendYield = 0.00;
OptSpec = 'call';
V0 = 0.04;
ThetaV = 0.05;
Kappa = 1;
SigmaV = 0.3;
RhoSV = -0.5;

formatIn = 'dd/mm/yyyy';
settle = datenum( '26/01/2021' , formatIn );
mat1 = datenum('19/03/2021', formatIn );
mat2 = datenum('16/04/2021', formatIn );
mat3 = datenum('21/05/2021', formatIn );
mat4 = datenum('18/06/2021', formatIn );
mat5 = datenum('17/09/2021', formatIn );
maturity = [mat1,mat2,mat3,mat4,mat5];
Maturity = (maturity - settle)/365;
z=[V0,ThetaV,Kappa,SigmaV,RhoSV];
% implied vol
vol_r = zeros(5,7);
call_heston = zeros(5,7);
for i=1:7
    for j=1:5
        vol_r(j,i) = blsimpv(AssetPrice, strike(i),0.0,Maturity(j), real(j,i));
        call_heston(j,i)= optByHestonFFT(Rate,AssetPrice,settle,maturity(j),OptSpec,strike(i),V0,ThetaV,Kappa,SigmaV,RhoSV);
    end
end
Error = abs(real - call_heston);
t = table(real,call_heston,Error);
writetable(t,'error.xlsx');

%Genetic Algo
% tic
% rng default
% numberOfVariables = 5;
% lb = [ 0.01,0.01,0.01,0.01,-1];
% ub = [10,2,10,10,1];
% objfun =@(x)HestonObjFunction(x);
% opts = optimoptions(@ga,'PlotFcn',{@gaplotbestf,@gaplotstopping},'MaxGenerations',120);
% opts.PopulationSize = 100;
% [p_ga,Fval_ga, exitFlag_ga, Output_ga]= ga(objfun, numberOfVariables,[],[],[],[],lb,ub,[],opts);
% result = fmincon(objfun, p_ga, [],[],[],[],lb, ub,[]);
% toc
   result = [0.185143210498775,0.169177294865401,3.63999916718472,0.575941612534588,-0.999516133374009];
price_cali = zeros(5,7);
vol_imp_cal = zeros(5,7);
for i = 1:7
    for j=1:5
        price_cali(j,i)= optByHestonFFT(Rate,AssetPrice,settle,maturity(j),OptSpec,strike(i),result(1),result(2),result(3),result(4),result(5));
        vol_imp_cal(j,i) = blsimpv(AssetPrice, strike(i),0.0,Maturity(j), price_cali(j,i));
    end
end
ga_iv_error =  abs(vol_r - vol_imp_cal).^2;
ga_error_price =  abs(real - price_cali);
t_1 = table((ga_error_price));
t_2 = table((ga_iv_error));
writetable(t_1,'ga_price_error.xlsx');
writetable(t_2,'ga_iv_error.xlsx');
ga = abs(real - price_cali).^2;
MSE_ga = sum(ga_iv_error(:))/numel(vol_r);


%LEVENBERG-MANQUARD
tic
lb = [ 0.01,0.01,0.01,0.01,-1];
ub = [10,2,10,10,1];
x0=[0.04, 0.05,1,0.3,-0.5];
objfun =@(x)HestonObjFunction_1(x);
options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt','MaxFunctionEvaluation',1500);
x = lsqnonlin(objfun,x0,lb,ub,options);
toc
% x = [0.185139575307730,0.169171845735356,3.63956551167195,0.575622871107371,-0.999999999246454];
f1 = 2*x(3)*x(2);
f2 = x(4).^2;
 
price_lm = zeros(5,7);
vol_imp_lm = zeros(5,7);
for i = 1:7
    for j=1:5
        price_lm(j,i)= optByHestonFFT(Rate,AssetPrice,settle,maturity(j),OptSpec,strike(i),x(1),x(2),x(3),x(4),x(5));
        vol_imp_lm(j,i) = blsimpv(AssetPrice, strike(i),0.0,Maturity(j), price_lm(j,i));
    end
end
lm_iv_error =  abs(vol_r - vol_imp_lm).^2;
lm_price_error =  abs(real - price_lm);
t_3 = table((lm_price_error));
t_4 = table((lm_iv_error));
writetable(t_3,'lm_price_error.xlsx');
writetable(t_4,'lm_iv_error.xlsx');
lm = abs(real - price_lm).^2;
MSE_lm = sum(lm_iv_error(:))/numel(vol_r);



%Surrogate
rng default % for reproducibility
tic
lb = [ 0.01,0.01,0.01,0.01,-1];
ub = [10,2,10,10,1];

objfun =@(x)HestonObjFunction(x); % objective
opts = optimoptions('surrogateopt','PlotFcn',[]);
[xsur_1,fsur,flgsur,osur] = surrogateopt(objfun,lb,ub,opts);
xsur = fmincon(objfun, xsur_1, [],[],[],[],lb, ub,[]);
toc
% xsur = [0.185142479517688,0.169176216603031,3.63991484143612,0.575878029877144,-0.999612802682312];
price_sur = zeros(5,7);
vol_imp_sur = zeros(5,7);
for i = 1:7
    for j=1:5
        price_sur(j,i)= optByHestonFFT(Rate,AssetPrice,settle,maturity(j),OptSpec,strike(i),xsur(1),xsur(2),xsur(3),xsur(4),xsur(5));
        vol_imp_sur(j,i) = blsimpv(AssetPrice, strike(i),0.0,Maturity(j), price_sur(j,i));
    end
end
sur_iv_error =  abs(vol_r - vol_imp_sur).^2;
sur_price_error =  abs(real - price_sur);
t_5 = table((sur_price_error));
t_6 = table((sur_iv_error));
writetable(t_5,'sur_price_error.xlsx');
writetable(t_6,'sur_iv_error.xlsx');
sur = abs(real - price_sur).^2;
MSE_sur = sum(sur_iv_error(:))/numel(vol_r);


%PLOTS
figure(1)    
surf(strike,Maturity,real)
xlabel('Strike')
ylabel('Maturity')
zlabel('price')
title(' REAL CALL PRICE');

figure(2)
surf(strike,Maturity,vol_r);
xlabel('Strike')
ylabel('Maturity')
zlabel('IV')
title(' B&S IMPLIED VOLATILITY');

figure(3)
surf(strike,Maturity,call_heston,'FaceColor','#A2142F')
hold on
surf(strike,Maturity,real,'FaceColor','#7E2F8E')
xlabel('Strike')
ylabel('Maturity')
zlabel('Price')
title('REAL VS. HESTON (ARBITRARY PARAMETERES)')
legend('Real','Heston')
hold off

figure(4)
surf(strike,Maturity,price_cali,'FaceColor','m')
hold on
surf(strike,Maturity,real,'FaceColor','c')
legend('GA CALIBRATED','REAL')
xlabel('Strike')
ylabel('Maturity')
zlabel('Price')
title('REAL PRICE VS GA CALIBRATION')
hold off


%GA CALIBRATION
figure(5)
surf(strike,Maturity,vol_imp_cal,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('IV')
title('IV - GA ')

figure(6)
surf(strike,Maturity,ga_iv_error,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('Error')
title('IV ERROR - GA ')

figure(7)
surf(strike,Maturity,ga_error_price,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('Error')
title('PRICE ERROR - GA')

figure(8)
bar3(transpose(ga_iv_error))
set(gca,'XTickLabel',[0.14 0.21 0.31 0.39 0.64])
set(gca,'YTickLabel',[265 270 275 280 285 290 295])
xlabel('Maturity')
ylabel('Strike')
zlabel('Error')
title('IV ERROR - GA')

%LM CALIBRATION
figure(9)
surf(strike,Maturity,price_lm,'FaceColor','g')
hold on
surf(strike,Maturity,price_cali,'FaceColor','m')
surf(strike,Maturity,real,'FaceColor','c')
legend('L-M','GA','REAL')
xlabel('Strike')
ylabel('Maturity')
zlabel('Price')
title('REAL - L-M - GA PRICE')
hold off

figure(10)
surf(strike,Maturity,vol_imp_lm,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('IV')
title('IV L-M')

figure(11)
surf(strike,Maturity,lm_iv_error,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('Error')
title('IV ERROR - L-M ')

figure(12)
surf(strike,Maturity,lm_price_error,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('Error')
title('PRICE ERROR - L-M')

figure(13)
bar3(transpose(lm_iv_error))
set(gca,'XTickLabel',[0.14 0.21 0.31 0.39 0.64])
set(gca,'YTickLabel',[265 270 275 280 285 290 295])
xlabel('Maturity')
ylabel('Strike')
zlabel('Error')
title('IV ERROR - L-M')

%SUR CALIBRATION
figure(14)
surf(strike,Maturity,vol_imp_sur,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('IV')
title('IV - SUR ')

figure(15)
surf(strike,Maturity,sur_iv_error,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('Error')
title('IV ERROR - SUR')

figure(16)
surf(strike,Maturity,sur_price_error,'FaceAlpha',0.7)
xlabel('Strike')
ylabel('Maturity')
zlabel('Error')
title(' ERROR PRICE - SUR')

figure(17)
bar3(transpose(sur_iv_error))
set(gca,'XTickLabel',[0.14 0.21 0.31 0.39 0.64])
set(gca,'YTickLabel',[265 270 275 280 285 290 295])
xlabel('Maturity')
ylabel('Strike')
zlabel('Error')
title('IV ERROR - SUR')

function MSE = HestonObjFunction (p)
    global  Rate AssetPrice maturity OptSpec strike real settle 
    Price_tmp = zeros(5,7);
    for i = 1:7
        for j=1:5
            Price_tmp(j,i) = optByHestonFFT(Rate,AssetPrice,settle,maturity(j),OptSpec,strike(i),p(1),p(2),p(3),p(4),p(5));
        end
    end
    er = abs(real - Price_tmp).^2;
    MSE = sum(er(:))/numel(real);
end
%For LM
function MSE = HestonObjFunction_1 (p)
    global  Rate AssetPrice maturity OptSpec strike real settle 
    Price_tmp = zeros(5,7);
    for i = 1:7
        for j=1:5
            Price_tmp(j,i) = optByHestonFFT(Rate,AssetPrice,settle,maturity(j),OptSpec,strike(i),p(1),p(2),p(3),p(4),p(5));
        end
    end
    
    %use this one for sqlnonlin
    MSE = abs(real - Price_tmp);
    
end


