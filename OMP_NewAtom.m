function para_Xnew = OMP_NewAtom(D, Y, L, errorGoal,para_Xold,d_k_t)

for k = 1:size(Y,2)
    y = Y(:,k);
        
    x_t = zeros(size(D,2)+1,1);
    x_t(1:size(D,2)) = para_Xold.X{k};

    select_ind = para_Xold.Sel_ind{k};
    
    r_t = para_Xold.r{k};

    Q_D_t= zeros(size(D',1)+1,size(D',2));
    Q_D_t(1:size(D'),:) = para_Xold.Q{k};
    
    iter = length(find(select_ind))+1;
    prev_range = 1:(iter-1);
    new_ind = size(D,2)+1;
    
    % update recursion variables
    b_tm1 = Q_D_t(prev_range, :) * d_k_t;
    if (iter == 1)
        % no atoms selected yet (initialization)
        d_tilde = d_k_t;
        q_t = d_tilde / (d_tilde' * d_tilde);
        Q_D_t(iter, :) = d_k_t' / (d_k_t' * d_k_t);
    else
        d_tilde = d_k_t - (D(:, select_ind(prev_range)) * b_tm1);
        q_t = d_tilde / (d_tilde' * d_tilde);
        Q_D_t(prev_range, :) = Q_D_t(prev_range, :) - (b_tm1 * q_t');
        Q_D_t(iter, :) = q_t';
    end
    
    % update coefficient vector
    alpha_t = q_t' * y;
    x_t(select_ind(prev_range)) = x_t(select_ind(prev_range)) - (alpha_t * b_tm1);
    select_ind(iter) = new_ind;
    x_t(new_ind) = alpha_t;
    
    % update residual
    r_t = r_t - (alpha_t * d_tilde);
    
    para_Xnew.X{k} = x_t;
    para_Xnew.Q{k} = Q_D_t;
    para_Xnew.Sel_ind{k} = select_ind;
    para_Xnew.r{k} = r_t;
    
end;

