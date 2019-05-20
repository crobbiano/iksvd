function X = RecursiveOMP(D,X,Y,ResidNorm)

% Specify coefficient tolerance
CoeffTol = 1e-4;

% Initialize coefficients
if isempty(X)
    X = zeros(size(D,2),size(Y,2));
elseif size(X,1) < size(D,2)
    X = [X;zeros(size(D,2)-size(X,1),size(X,2))];
end

for i = 1:size(Y,2)
    
    % Initialize variables
    y = Y(:,i);
    resid_norm = ResidNorm*norm(y);
    Indices = find(X(:,i)~=0);
    
    if isempty(Indices)
        
        % Initialize variables
        r = y;
        Dt = [];
        
        % Find best match
        [~,ind] = max(abs(r'*D));
        Dt = [Dt,D(:,ind)];
        Indices = [Indices;ind];
        
        % Update filters
        Q = Dt'/(Dt'*Dt);
        alpha = Q*y;
        
        % Update coefficients
        x = alpha;
        r = r-alpha*Dt;
        
    else
        
        % Build dictionary and filter
        Dt = D(:,Indices);
        Q = inv(Dt'*Dt)*Dt';
        
        % Compute coefficients and residual
        x = Q*y;
        r = y-Dt*x;
        
    end
    
    % Add atoms in D until norm of residual falls below specified value
    xbest = x;
    maxNumIterations = 100;
    numIterations = 1;
    rprev = r;
    Indicesbest = [];
    while (norm(r) > resid_norm) & (numIterations < maxNumIterations)
                
        % Find best match
        [~,ind] = max(abs(r'*D));
        d = D(:,ind);
        Indices = [Indices;ind];
        
        % Update filters
        b = Q*d;
        d_tilde = d-Dt*b;
        q = d_tilde/(norm(d_tilde)^2);
        alpha = q'*y;
        Dt = [Dt,d];
        Q = [Q-b*q';q'];
        
        % Update coefficients
        x = [x-alpha*b;alpha];
        r = r-alpha*d_tilde;
        
        if norm(rprev) > norm(r)
            xbest = x;
            Indicesbest = Indices;
        end
        rprev = r;
        numIterations = numIterations + 1;
    end
    if isempty(Indicesbest)
        Indicesbest = Indices;
        xbest = x;
%         display(['fail']);
    end
    % Fill-in coefficients
    X(Indicesbest,i) = xbest;
%     X(Indices,i) = x;
    
end

% Null coefficients below specified tolerance
X(abs(X) < CoeffTol) = 0;