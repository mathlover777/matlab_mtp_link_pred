function shorten_graph( data_set_name)
    lom = load(strcat('datasets/',data_set_name,'_lom.mat'));
    X = load(strcat('datasets/',data_set_name,'.mat'));
    
    if isfield(X,'processed')
        fprintf('graph already processed');
        return
    end
    
    lom = transpose(sort(lom.lom));
    
    [~,n] = size(lom);
    disp(n);
    D = zeros(n,n);
    [feature_dim,~] = size(X.F);
    F = zeros(feature_dim,n);
    l = 1;
    for i = lom
        k = 1;
        for j = lom
            %disp(i);
          D(l,k) = X.D(i,j);
          k = k + 1;
        end
        l = l + 1;
    end
    l = 1;
    for i = 1:feature_dim
        k = 1;
        for j = lom
           F(l,k) = X.F(i,j);
           k = k + 1;
        end
        l = l + 1;
    end
    processed = 1;
    G = X.D;
    H = X.F;
    save(strcat('datasets/',data_set_name,'.mat'),'D','F','processed','G','H');

end

