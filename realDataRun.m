% use train-test split, or train on whole network?
USE_TRAIN_TEST_SPLIT = 1;

A = load('datasets/intl-conflict.mat');

% 1 = standard, 2 = ranking
for WHICH_OPTIMISATION = [ 1 2 ]

     % # of per-node explicit features
    for dBilinear = [ 3 ]
    
        % # of per-link explicit features
        for dPair = [ 5]
        
            fprintf('testing setting: ');
            if WHICH_OPTIMISATION == 1, fprintf('standard loss, '); else fprintf('ranking loss, '); end;
            fprintf('# node features = %d, ', dBilinear);
            fprintf('# link features = %d\n', dPair);
            
            
            
            [m,~] = size(A.D);
            k = 30; % # of latent features

            %% generate synthetic network based on factorisation model
            rng('default');

%             UTrue = 1/sqrt(k) * randn(k, m); % "true" latent features
%             STrue = UTrue' * UTrue; % "true" affinity scores between each i and j

            nodeFeaturesPresent = double(dBilinear > 0);
            linkFeaturesPresent = double(dPair > 0);

            % side-information for nodes
            if nodeFeaturesPresent    
                sideBilinear = A.F; % side information per node
                sideBilinear = bsxfun(@minus, sideBilinear, min(sideBilinear, [], 2)); % normalise features to [0, 1]
                sideBilinear = bsxfun(@rdivide, sideBilinear, max(sideBilinear, [], 2));

                WBilinearTrue = randn(dBilinear); % "true" explicit feature weights

%                 STrue = STrue + sideBilinear' * WBilinearTrue * sideBilinear;
            else
                sideBilinear = [];
            end

            % side-information for links
            if linkFeaturesPresent
                
%                 sidePair = randn(dPair, m, m); % side information per link
%                 for i = 1:m
%                     for j = 1:m
%                         sidePair(1,i,j) = A.similarityX(i,j);
%                     end
%                 end
                sidePair = A.X;
                sidePair = bsxfun(@minus, sidePair, min(reshape(sidePair, [dPair m*m]), [], 2)); % normalise features to [0, 1]
                sidePair = bsxfun(@rdivide, sidePair, max(reshape(sidePair, [dPair m*m]), [], 2));

                WPairTrue = randn(1, dPair); % "true" explicit feature weights    

%                 STrue = STrue + reshape(WPairTrue * reshape(sidePair, [dPair m*m]), [m m]);
            else
                sidePair = [];
            end
%             disp(STrue);
            
%             PTrue = 1./(1 + exp(-STrue)); % prob of link between i and j
%             disp(PTrue);
            
            
%             GTrue = 2 * double(rand(size(PTrue)) <= PTrue) - 1; % threshold to give binary label
%             disp(GTrue);
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
%             return;
            for i = 1:m
                for j=1:m
                    if GTrue(i,j) == 0
                        GTrue(i,j) = -1;
                    end
                end
            end
            [i,j,v] = find(GTrue);
            disp(size([i,j,v]));
%             disp([i,j,v]);
            
%             return;
            
            D = [ i j (v + 1)/2 ]'; % labels must be { 0,1 }
    
%             disp(D);
%             return;
            
            count = 0;
%             Pairs = [];
            for a = 1:m
                for b = 1:m
                    for c = 1:m
                        if GTrue(a,b) == 1 && GTrue(a,c) == -1
                            count = count + 1;
%                             Pairs = [ Pairs [ a; b; c ] ];
                        end
                    end
                end
            end
%             disp(count);
%             disp(size(Pairs));
%             disp(Pairs);
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
%             disp(Pairs)
%             disp(size(Pairs));
            
%             return;

            if USE_TRAIN_TEST_SPLIT                
                TRAIN_RATIO = 0.1;
%                 size_ss = size(D,2);
%                 disp(size_ss);
%                 return;
                I = randperm(size(D,2));
                ITr = I(1:ceil(TRAIN_RATIO * length(I)));
                ITe = I(1+ceil(TRAIN_RATIO * length(I)):end);

                DTr = D(:,ITr);
                DTe = D(:,ITe);
                
                disp(size(DTr));
                disp(size(DTe));
                
%                 return;
                
                % for pairs one it is more correct to first subsample the network
                % and then pick pairs from that
                % this approach is just for illustration
                J = randperm(size(Pairs,2));
                JTr = J(1:ceil(TRAIN_RATIO * length(J)));
                JTe = J(1+ceil(TRAIN_RATIO * length(J)):end);

                PairsTr = Pairs(:,JTr);
                PairsTe = Pairs(:,JTe);
            else
                % use all links
                ITr = 1:m*m; ITe = 1:m*m;
                JTr = 1:size(Pairs,2); JTe = 1:size(Pairs,2);
                DTr = D(:,ITr);
                DTe = D(:,ITe);
                PairsTr = Pairs(:,JTr);
                PairsTe = Pairs(:,JTe);
            end

            %% initialise parameters

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

            eta = [];

            % potentially need different learning rates, based on the
            % type of optimisation used
            if WHICH_OPTIMISATION == 1
                eta.etaLatent = 0.1; % learning rate for latent feature
                eta.etaRowBias = 0.01; % learning rate for node bias
                eta.etaLatentScaler = 0.1; % learning rate for scaler to latent features

                eta.etaPair = linkFeaturesPresent * 1;
                eta.etaBilinear = nodeFeaturesPresent * 1;

                eta.etaBias = linkFeaturesPresent * 1; % learning rate for global bias, used when features are present
                
                EPOCHS = 25; % # of passes of SGD
            else
                eta.etaLatent = 0.1; % learning rate for latent feature
                eta.etaRowBias = 0.01; % learning rate for node bias
                eta.etaLatentScaler = 0.1; % learning rate for scaler to latent features

                eta.etaPair = linkFeaturesPresent * 1;
                eta.etaBilinear = nodeFeaturesPresent * 1;

                eta.etaBias = linkFeaturesPresent * 1e-4; % learning rate for global bias, used when features are present
                
                EPOCHS = 1; % # of passes of SGD
            end

            epochFrac = 1; % fraction of +'ve and -'ve pairs to use in each pass
            batchSize = 1; % # of examples in each update

            loss = 'log';
            link = 'sigmoid';

            symmetric = 0; % network is symmetric?

            %% learn matrix factorisation model with no side-information
            if WHICH_OPTIMISATION == 1
                [U, UBias, ULatentScaler, WPair, WBias, WBilinear] = factorizationSideInfoJointSGDOptimizer(DTr, sidePair, sideBilinear, weights, lambda, eta, EPOCHS, epochFrac, [], [], loss, link, symmetric, { [] });
            else
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
%             disp(size(PPred));
%             return;
            %% print error statistics
            
%             disp(size(v));
%             return;
            
            testLinks = sub2ind(size(PPred), DTe(1,:), DTe(2,:));
%             fprintf('rmse of predicted vs true probabilities = %1.4f\n', sqrt(mean((PPred(testLinks) - PTrue(testLinks)).^2)));
%             disp((v(ITe)+1)/2)
%             disp(PPred(testLinks));
                disp(size(v(ITe)));
                disp(size(PPred(testLinks)));
%                 return;
%             [~,~,~,trueAuc] = perfcurve((v(ITe)+1)/2, PTrue(testLinks), 1);
%             fprintf('optimal auc = %1.4f\n', trueAuc);
%             disp([(v(ITe)+1)/2 transpose(PPred(testLinks))]);
            [X,Y,T,myAuc] = perfcurve((v(ITe)+1)/2, transpose(PPred(testLinks)), 1);
            fprintf('predicted auc = %1.4f\n', myAuc);
            [X,Y,T,myAuc] = perfcurve((v(ITe)+1)/2, PPred(testLinks), 1);
            fprintf('predicted auc actual = %1.4f\n', myAuc);
%             return;
%             plot(X,Y);
            fprintf('\n');
        end
    end
end
