function model = processDynamic(data, model, numpoles, doHyst)
  global bestcost

  options=optimset('TolX',1e-8,'TolFun',1e-8,'MaxFunEval',100000, ...
    'MaxIter',1e6,'Jacobian','Off'); 
  options=optimset('TolX',0.1,'TolFun',1e-2,'MaxFunEval',40, ...
    'MaxIter',20,'Jacobian','Off'); 
    
  % Step 1: Compute capacity and coulombic efficiency 
  totDisAh = data.script1.disAh(end) + data.script2.disAh(end) + data.script3.disAh(end);
  totChgAh = data.script1.chgAh(end) + data.script2.chgAh(end) + data.script3.chgAh(end);
             
  eta25 = totDisAh/totChgAh; 
  data.eta = eta25; 
  
  data.script1.chgAh = data.script1.chgAh*eta25;
  data.script2.chgAh = data.script2.chgAh*eta25;
  data.script3.chgAh = data.script3.chgAh*eta25;    

  Q25 = data.script1.disAh(end) + data.script2.disAh(end) - data.script1.chgAh(end) - data.script2.chgAh(end);
  data.Q = Q25; 
  
  model.temps    = 25; 
  model.etaParam = eta25;
  model.QParam   = Q25;
  
  % Step 2: Compute OCV for "discharge portion" of test
  etaik = data.script1.current(:); 
  etaik(etaik<0)= eta25*etaik(etaik<0);
  data.Z = 1 - cumsum([0; etaik(1:end-1)]) * 1/(Q25*3600); 
  data.OCV = OCVfromSOCtemp(data.Z(:), 25, model);
  
  % Step 3: Now, optimize!
  model.GParam  = NaN; 
  model.M0Param = NaN; 
  model.MParam  = NaN; 
  model.R0Param = NaN; 
  model.RCParam = NaN(1,numpoles); 
  model.RParam  = NaN(1,numpoles); 

  fprintf('Processing temperature 25°C\n');
  bestcost = Inf;
  
  if doHyst
    model.GParam = abs(fminbnd(@(x) optfn(x,data,model,25,doHyst),1,250,options));
  else
    model.GParam = 0;
    optfn(0,data,model,25,doHyst);
  end
  [~,model] = minfn(data,model,25,doHyst);                          
end

% ====================================================================
% HELPER FUNCTIONS (Perfectly safe inside a .m file)
% ====================================================================

function value = getParamESC(paramName, temperature, model)
    tempIndex = find(model.temps == temperature);
    if isempty(tempIndex)
        error('Temperature %d not found in the model.', temperature);
    end
    if strcmp(paramName, 'RCParam') || strcmp(paramName, 'RParam')
        value = model.(paramName)(tempIndex, :);
    else
        value = model.(paramName)(tempIndex);
    end
end

function cost = optfn(theGParam,data,model,theTemp,doHyst)
  global bestcost 
  model.GParam = abs(theGParam);
  [cost,model] = minfn(data,model,theTemp,doHyst);
  if cost < bestcost 
    bestcost = cost;
    disp('    The model created for this value of gamma is the best ESC model yet!');
  end
end

function [cost,model] = minfn(data,model,theTemp,doHyst)
  G = abs(getParamESC('GParam',25,model));
  Q = abs(getParamESC('QParam',25,model));
  eta = abs(getParamESC('etaParam',25,model));
  RC = getParamESC('RCParam',25,model);
  numpoles = length(RC);
  
  ik = data.script1.current(:);
  vk = data.script1.voltage(:);
  tk = (1:length(vk))-1;
  etaik = ik; etaik(ik<0) = etaik(ik<0)*eta;

  h = 0*ik; sik = 0*ik;
  fac = exp(-abs(G*etaik/(3600*Q)));
  for k = 2:length(ik)
    h(k) = fac(k-1)*h(k-1) + (fac(k-1)-1)*sign(ik(k-1));
    sik(k) = sign(ik(k));
    if abs(ik(k)) < Q/100, sik(k) = sik(k-1); end
  end
  
  vest1 = data.OCV;
  verr = vk - vest1;
  
  np = numpoles; 
  max_np = numpoles + 5; % The safety limit we added!
  
  for current_np = np : max_np
    fprintf('    Trying np = %d...\n', current_np); fflush(stdout);
    A = SISOsubid(-diff(verr),diff(etaik),current_np);
    eigA = eig(A); 
    eigA = eigA(eigA == conj(eigA));  
    eigA = eigA(eigA > 0 & eigA < 1); 
    okpoles = length(eigA); 
    
    if okpoles >= numpoles
        np = current_np + 1; 
        break; 
    end
    if current_np == max_np
        error('Gave up! Could not find enough valid poles after trying up to np = %d.', max_np);
    end
  end
  
  RCfact = sort(eigA); RCfact = RCfact(end-numpoles+1:end);
  RC = -1./log(RCfact);
  
  vrcRaw = zeros(numpoles,length(h));
  for k = 2:length(ik)
    vrcRaw(:,k) = diag(RCfact)*vrcRaw(:,k-1) + (1-RCfact)*etaik(k-1);
  end
  vrcRaw = vrcRaw';

  if doHyst
    H = [h,sik,-etaik,-vrcRaw]; 
    W = lsqnonneg(H,verr); 
    M = W(1); M0 = W(2); R0 = W(3); Rfact = W(4:end)';
  else
    H = [-etaik,-vrcRaw]; 
    W = H\verr;    
    M=0; M0=0; R0 = W(1); Rfact = W(2:end)';
  end
  
  model.R0Param = R0;
  model.M0Param = M0;
  model.MParam = M;
  model.RCParam = RC';
  model.RParam = Rfact';
  
  vest2 = vest1 + M*h + M0*sik - R0*etaik - vrcRaw*Rfact';
  verr = vk - vest2;
      
  v1 = OCVfromSOCtemp(0.95,25,model);
  v2 = OCVfromSOCtemp(0.05,25,model);
  N1 = find(vk<v1,1,'first'); N2 = find(vk<v2,1,'first');
  if isempty(N1), N1=1; end; if isempty(N2), N2=length(verr); end
  
  cost=sqrt(mean(verr(N1:N2).^2)); 
  fprintf('  RMS error for present value of gamma = %0.2f (mV)\n',cost*1000);
  if isnan(cost), error('Cost evaluated to NaN. Stopping.'), end
end

function A = SISOsubid(y,u,n)
  y = y(:); y = y'; ny = length(y); 
  u = u(:); u = u'; nu = length(u); 
  i = 2*n; 
  twoi = 4*n;           
  j = ny-twoi+1;
  Y=zeros(twoi,j); U=zeros(twoi,j);
  for k=1:2*i
    Y(k,:)=y(k:k+j-1); U(k,:)=u(k:k+j-1);
  end
  R = triu(qr([U;Y]'))'; 
  R = R(1:4*i,1:4*i); 	 
  Rf = R(3*i+1:4*i,:);              
  Rp = [R(1:1*i,:);R(2*i+1:3*i,:)]; 
  Ru  = R(1*i+1:2*i,1:twoi); 	      
  Rfp = [Rf(:,1:twoi) - (Rf(:,1:twoi)/Ru)*Ru,Rf(:,twoi+1:4*i)]; 
  Rpp = [Rp(:,1:twoi) - (Rp(:,1:twoi)/Ru)*Ru,Rp(:,twoi+1:4*i)]; 
  if (norm(Rpp(:,3*i-2:3*i),'fro')) < 1e-10
    Ob = (Rfp*pinv(Rpp')')*Rp; 	
  else
    Ob = (Rfp/Rpp)*Rp;
  end
  WOW = [Ob(:,1:twoi) - (Ob(:,1:twoi)/Ru)*Ru,Ob(:,twoi+1:4*i)];
  [U,S,~] = svd(WOW);
  ss = diag(S);
  U1 = U(:,1:n); 
  gam  = U1*diag(sqrt(ss(1:n)));
  gamm = gam(1:(i-1),:);
  gam_inv  = pinv(gam); 			
  gamm_inv = pinv(gamm); 			
  Rhs = [gam_inv*R(3*i+1:4*i,1:3*i),zeros(n,1); R(i+1:twoi,1:3*i+1)];
  Lhs = [gamm_inv*R(3*i+1+1:4*i,1:3*i+1); R(3*i+1:3*i+1,1:3*i+1)];
  sol = Lhs/Rhs;    
  A = sol(1:n,1:n); 
end