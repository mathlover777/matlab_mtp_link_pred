data_mat_list = {'condmat','intl-conflict','metabolic','nips_1-17','powergrid','prot-prot'};

data_path_list = strcat('datasets/',data_mat_list,'.mat');
output_path_list = strcat(data_mat_list,'.mat');

bilinear_available = [false,true,true,false,false,true];

pair_available = [false,true,false,false,false,false];
is_symmetric = [true,true,true,true,true,true];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% running code for prot-prot
data_index = 6;
disp(data_path_list{data_index});
data_name = data_mat_list{data_index};
data_path = data_path_list{data_index};
A = load(data_path_list{data_index});

[m,~] = size(A.D);
% latent feature
k = 10;

% clean linear features if given
if bilinear_available(data_index)
    [dBilinear,~] = size(A.F);
    sideBilinear = A.F;
    sideBilinear = bsxfun(@minus, sideBilinear, min(sideBilinear, [], 2)); % normalise features to [0, 1]
    sideBilinear = bsxfun(@rdivide, sideBilinear, max(sideBilinear, [], 2));
    WBilinearTrue = randn(dBilinear);
else
    sideBilinear = [];
    dBilinear = 0;
end
% create dyad features if given
if pair_available(data_index)
    [dPair,~,~] = size(A.X);
    sidePair = A.X;
    sidePair = bsxfun(@minus, sidePair, min(reshape(sidePair, [dPair m*m]), [], 2)); % normalise features to [0, 1]
    sidePair = bsxfun(@rdivide, sidePair, max(reshape(sidePair, [dPair m*m]), [], 2));
    WPairTrue = randn(1, dPair);
else
    dPair = 0;
    sidePair = [];
end

nodeFeaturesPresent = double(dBilinear > 0);
linkFeaturesPresent = double(dPair > 0);


% cleaning the adjacency matrix
GTrue = A.D;
            
z_count = 0;
p_count = 0;
n_count = 0;

for i = 1:m
    for j=1:m
        if GTrue(i,j) == 1
            p_count = p_count + 1;
        elseif GTrue(i,j) == 0
            z_count = z_count + 1;
        elseif GTrue(i,j) == -1;
            n_count = n_count + 1;
        end
    end
end
disp([z_count,p_count,n_count,z_count+p_count+n_count,m*m]);
for i = 1:m
    for j=1:m
        if GTrue(i,j) == 0
            GTrue(i,j) = -1;
        end
    end
end
[i,j,v] = find(GTrue);
D = [ i j (v + 1)/2 ]';

count = 0;
for a = 1:m
    for b = 1:m
        for c = 1:m
            if GTrue(a,b) == 1 && GTrue(a,c) == -1
                count = count + 1;
            end
        end
    end
end
Pairs = zeros(3,count);
l = 1;
for a = 1:m
    for b = 1:m
        for c = 1:m
            if GTrue(a,b) == 1 && GTrue(a,c) == -1
                Pairs(:,l) = [ a; b; c ] ;
                l = l + 1;
            end
        end
    end
end
fprintf('done 3 loop');
%%%%%%%%%%%%%%%%%%%%%%%%% SPLIT THE DATASET %%%%%%%%%
TRAIN_RATIO = 0.4;
I = randperm(size(D,2)); % finding a permutation of [1 2 ... m^2]
ITr = I(1:ceil(TRAIN_RATIO * length(I))); % training indexes
ITe = I(1+ceil(TRAIN_RATIO * length(I)):end); % testing indexes
DTr = D(:,ITr); % train data
DTe = D(:,ITe); % test data
J = randperm(size(Pairs,2));
JTr = J(1:ceil(TRAIN_RATIO * length(J)));
JTe = J(1+ceil(TRAIN_RATIO * length(J)):end);
PairsTr = Pairs(:,JTr); % train pair
PairsTe = Pairs(:,JTe); % test pair

%%%%%%%%%%%%%%%%% setting the config values %%%%%%%%%%%%
weights = [];
weights.U = 1/sqrt(k) * randn(k, m); % for k latent features and m nodes
weights.UBias = randn(1, m);
weights.ULatentScaler = 1/sqrt(k) * randn(k, k); % for asymmetric; for symmetric, use diag(randn(k)) instead
weights.WPair = linkFeaturesPresent * randn(1, dPair); % for dPair features for each pair
weights.WBilinear =  nodeFeaturesPresent * randn(dBilinear); % for dBilinear features for each node
weights.WBias = linkFeaturesPresent * randn;
lambda = [];
lambda.lambdaLatent = 0.1; % regularization for node's latent vector U
lambda.lambdaRowBias = 0; % regularization for node's bias UBias
lambda.lambdaLatentScaler = 0; % regularization for scaling factors Lambda (in paper)
lambda.lambdaPair = 1e-5; % regularization for weights on pair features
lambda.lambdaBilinear = 1e-5; % regularization for weights on node features
lambda.lambdaScaler = 1; % scaling factor for regularization, can be set to 1 by default 

epochFrac = 1; % fraction of +'ve and -'ve pairs to use in each pass
batchSize = 1; % # of examples in each update
loss = 'log';
link = 'sigmoid';
symmetric = is_symmetric(data_index); % network is symmetric?
fprintf('Symmetric : %d',symmetric);
eta = [];

for WHICH_OPTIMISATION = [ 1 2 ]
    if WHICH_OPTIMISATION == 1
        % %%%%%%%%%% learning rates for factorizationSideInfoJointSGDOptimizer
        eta.etaLatent = 0.1; % learning rate for latent feature
        eta.etaRowBias = 0.01; % learning rate for node bias
        eta.etaLatentScaler = 0.1; % learning rate for scaler to latent features
        eta.etaPair = linkFeaturesPresent * 1;
        eta.etaBilinear = nodeFeaturesPresent * 1;
        eta.etaBias = linkFeaturesPresent * 1; % learning rate for global bias, used when features are present
        EPOCHS = 25; % # of passes of SGD
        [U, UBias, ULatentScaler, WPair, WBias, WBilinear] = factorizationSideInfoJointSGDOptimizer(DTr, sidePair, sideBilinear, weights, lambda, eta, EPOCHS, epochFrac, [], [], loss, link, symmetric, { [] });
    else
        % %%%%%%%%%% learning rates factorizationSideInfoRankingJointSGDOptimizer
        eta.etaLatent = 0.1; % learning rate for latent feature
        eta.etaRowBias = 0.01; % learning rate for node bias
        eta.etaLatentScaler = 0.1; % learning rate for scaler to latent features
        eta.etaPair = linkFeaturesPresent * 1;
        eta.etaBilinear = nodeFeaturesPresent * 1;
        eta.etaBias = linkFeaturesPresent * 1e-4; % learning rate for global bias, used when features are present
        EPOCHS = 1; % # of passes of SGD
        [U, UBias, ULatentScaler, WPair, WBias, WBilinear] = factorizationSideInfoRankingJointSGDOptimizer(PairsTr, sidePair, sideBilinear, weights, lambda, eta, EPOCHS, epochFrac, batchSize, loss, [], [], symmetric, [], { [] });
    end
    SPred = bsxfun(@plus, U * ULatentScaler * U', bsxfun(@plus, UBias', UBias)); % predicted score for (i, j) = U(i) * L * U(j)' + UBias(i) + UBias(j)
    
    if nodeFeaturesPresent
        SPred = SPred + sideBilinear' * WBilinear * sideBilinear + WBias;
    end

    if linkFeaturesPresent
        SPred = SPred + reshape(WPair * reshape(sidePair, [dPair m*m]), [m m]);
    end

    PPred = 1./(1 + exp(-SPred)); % predicted probability

    %%%%%%%%%%%%%% analyzing results %%%%%%%%%%%%%%%%
    
    
    analyze_caller = analyze;
    results = analyze_caller.get_results(PPred,GTrue,v,DTe,ITe);
    
    recall = results.recall;
    precision = results.precision;
    
    fprintf('\nRecall = %1.4f Precision = %1.4f\n',recall,precision);

    data_to_save = [];
    data_to_save.results = results;
    data_to_save.ITe = ITe;
    data_to_save.PPred = PPred;
    data_to_save.v = v;
    data_to_save.DTe = DTe;
    data_to_save.GTrue = GTrue;
    
    save(strcat('output/',int2str(WHICH_OPTIMISATION),'_',output_path_list{data_index}),'data_to_save');
    
    

end

