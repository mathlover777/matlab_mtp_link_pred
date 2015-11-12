% Stochastic gradient optimization of latent features + side-information using ranking
% ind = pairs x 3, where each row = (node 1, node 2, node 3), such that (node1, node2) has a positive link but (node1, node3) has a negative link
% sidePair = dPair x |V| x |V|, |V| = # of nodes; features for each pair of nodes
% sideBilinear = dBilinear x |V|; features for each node
% weights = structure with all the learned weights inside it
% lambda = structure with all the regularization parameters inside it
% eta = structure with all the learning rates inside it
% EPOCHS = # of sweeps over D
% epochFrac = used if we only want to do a partial sweep over D
% batchSize = # of examples to use in each stochastic gradient update
% loss = {'square', 'logistic'}
% convergenceScore{Tr,Te} = function that computes error on train/test set
% symmetric = {0, 1}; is the graph symmetric?
% mahalanobis = use special mahalanobis form?
% covarianceFull = precomputed values of sideBilinear(:,i)*sideBilinear(:,j)'
function [U, UBias, ULatentScaler, WPair, WBias, WBilinear, trainError, testError] = factorizationSideInfoRankingJointSGDOptimizer(ind, sidePair, sideBilinear, weights, lambda, eta, EPOCHS, epochFrac, batchSize, loss, convergenceScoreTr, convergenceScore, symmetric, mahalanobis, covarianceFull)

    pairs = size(ind, 2);
    %fprintf('processing %d pairs (1e%d)', pairs, round(log10(pairs)));

    U = weights.U; UBias = weights.UBias; ULatentScaler = weights.ULatentScaler; WPair = weights.WPair; WBias = weights.WBias; WBilinear = weights.WBilinear;
    lambdaLatent = lambda.lambdaLatent; lambdaRowBias = lambda.lambdaRowBias; lambdaPair = lambda.lambdaPair; lambdaBilinear = lambda.lambdaBilinear; lambdaLatentScaler = lambda.lambdaLatentScaler;
    etaLatent = eta.etaLatent; etaLatentScaler0 = eta.etaLatentScaler; etaRowBias0 = eta.etaRowBias; etaPair0 = eta.etaPair; etaBilinear0 = eta.etaBilinear; etaBias0 = eta.etaBias;    
    
    square = strcmp(loss,'square');
    lowRank = (size(WBilinear,1) ~= size(WBilinear,2));
    covarianceCached = (numel(covarianceFull{:}) > 0);
    hasDyadicSideInfo = numel(sidePair) > 0;
    
    lastScore = 0; trainError = []; testError = [];

    UOld = U; UBiasOld = UBias;
    WPairOld = WPair; WBiasOld = WBias;
    WBilinearOld = WBilinear;

    if numel(convergenceScore) > 0
        if symmetric
            trainError = convergenceScoreTr(U',UBias,ULatentScaler,WPair,WBias,WBilinear+WBilinear');
            testError = convergenceScore(U',UBias,ULatentScaler,WPair,WBias,WBilinear+WBilinear'); lastScore = testError.auc;
        else
            trainError = convergenceScoreTr(U',UBias,ULatentScaler,WPair,WBias,WBilinear);
            testError = convergenceScore(U',UBias,ULatentScaler,WPair,WBias,WBilinear); lastScore = testError.auc;
        end
        mse = 0; %computeMSE(obs,ind,x,W,y,lowRank,offset);
        disp(sprintf('original auc %.4g, mse %.4g',lastScore,mse));
    else
        lastScore = 0;
    end

    for e = 1 : EPOCHS
        etaU = etaLatent/((1 + etaLatent*lambdaLatent)*e);        
        etaRowBias = etaRowBias0/((1 + etaRowBias0*lambdaRowBias)*e);
        etaLatentScaler = etaLatentScaler0/((1 + etaLatentScaler0*lambdaLatentScaler)*e);
        etaBias = etaBias0/((1 + etaBias0*lambdaLatent)*e);
        etaPair = etaPair0/((1 + etaPair0*lambdaPair)*e);
        etaBilinear = etaBilinear0/((1 + etaBilinear0*lambdaBilinear)*e);        

        % Random shuffle of the training set
        I = randperm(pairs);
        ind = ind(:,I);

        for t = 1 : round(epochFrac * pairs)

            examples = (t-1)*batchSize+1:min(pairs,t*batchSize);
            examplesU = ind(1,examples); examplesI = ind(2,examples); examplesJ = ind(3,examples);

            %% Prediction

            % Latent part
            prediction = U(:,examplesU)' * ULatentScaler * (U(:,examplesI) - U(:,examplesJ)) + UBias(examplesI) - UBias(examplesJ);

            % Side-info part
            if etaBilinear > 0
                if lowRank
                    prediction = prediction + diag((sideBilinear(:,examplesU)' * (WBilinear'*WBilinear) * (sideBilinear(:,examplesI)) - sideBilinear(:,examplesJ))); % E x 1
                else
                    if covarianceCached
                        covariance = covarianceFull{examplesU,examplesI} - covarianceFull{examplesU,examplesJ};
                    else
                        covariance = sideBilinear(:,examplesU) * (sideBilinear(:,examplesI) - sideBilinear(:,examplesJ))';
                    end                    

                    if mahalanobis
                        predictionIJ = exp(-(sideBilinear(:,examplesU)'*WBilinear*sideBilinear(:,examplesU) + sideBilinear(:,examplesI)'*WBilinear*sideBilinear(:,examplesI) - 2*sideBilinear(:,examplesU)'*WBilinear*sideBilinear(:,examplesI)));
                        predictionIK = exp(-(sideBilinear(:,examplesU)'*WBilinear*sideBilinear(:,examplesU) + sideBilinear(:,examplesJ)'*WBilinear*sideBilinear(:,examplesJ) - 2*sideBilinear(:,examplesU)'*WBilinear*sideBilinear(:,examplesJ)));
                        prediction = prediction + predictionIJ - predictionIK;
                    else
                        prediction = prediction + diag((sum(sum(WBilinear .* covariance)))); % E x 1
                        if symmetric
                            prediction = prediction + sum(sum(WBilinear .* covariance'));
                        end
                    end
                end
            end

            if hasDyadicSideInfo
                prediction = prediction + WPair * (sidePair(:,examplesU,examplesI) - sidePair(:,examplesU,examplesJ));            
                prediction = prediction + WBias;
            end

            %% Gradients
            if ~square
                prediction = 1./(1 + exp(-prediction));
                gradScaler = (prediction - 1); % 1 x 1
            else
                gradScaler = 2*(prediction - 1); % 1 x 1
            end

            gradU = ULatentScaler * (U(:,examplesI) - U(:,examplesJ)); % d x 1
            gradI = ULatentScaler' * U(:,examplesU);
            gradJ = -ULatentScaler' * U(:,examplesU);
            gradUBias = ones(1,numel(examples)); % 1 x 1

            if etaLatentScaler > 0
                if symmetric
                    gradLatentScaler = diag(U(:,examplesU) .* (U(:,examplesI) - U(:,examplesJ)));
                else
                    gradLatentScaler = U(:,examplesU) * (U(:,examplesI) - U(:,examplesJ))';
                end
            end            
            
            gradUL = lambdaLatent * U(:,examplesU);
            gradIL = lambdaLatent * U(:,examplesI);
            gradJL = lambdaLatent * U(:,examplesJ);

            if etaBilinear > 0
                if lowRank
                    gradBilinear = (WBilinear * (covariance + covariance'));
                else                
                    if mahalanobis
                        gradIJ = predictionIJ * (2*covariance{examplesU,examplesI} - covariance{examplesU,examplesU} - covariance{examplesI,examplesI});
                        gradIK = predictionIK * (2*covariance{examplesU,examplesJ} - covariance{examplesU,examplesU} - covariance{examplesJ,examplesJ});
                        gradBilinear = gradIJ - gradIK;
                    else
                        gradBilinear = covariance; % d x d 
                    end
                end
            end

            if hasDyadicSideInfo
                gradPair = (sidePair(:,examplesU,examplesI) - sidePair(:,examplesU,examplesJ))';
                gradBias = ones(1,numel(examples));
            end

             if symmetric                 
                 if etaBilinear > 0, gradBilinear = gradBilinear + gradBilinear'; end;
             end

            gradPairL = lambdaPair * WPair;

            %% Updates
            U(:,[examplesU examplesI examplesJ]) = U(:,[examplesU examplesI examplesJ]) - etaU/batchSize * (gradScaler * [gradU gradI gradJ] + [gradUL gradIL gradJL]);
            UBias([examplesI; examplesJ]) = UBias([examplesI; examplesJ]) - etaRowBias/batchSize * (gradScaler * [gradUBias -gradUBias]);

            if etaLatentScaler > 0
                ULatentScaler = ULatentScaler - etaLatentScaler/batchSize * (gradScaler * gradLatentScaler + lambdaLatentScaler * ULatentScaler);
            end            
            
            if etaBilinear > 0 % avoids wasteful computation...
                WBilinear = WBilinear - etaBilinear/batchSize * (gradScaler * gradBilinear + lambdaBilinear * WBilinear);
            end
            
            if hasDyadicSideInfo
                WPair = WPair - etaPair/batchSize * (gradScaler * gradPair + gradPairL);
                WBias = WBias - etaBias/batchSize * (gradScaler * gradBias);
            end
        end

        % Periodic information about model performance        
        if numel(convergenceScoreTr) > 0 && mod(e-1,5) == 0        
            if symmetric
                testError = convergenceScore(U',UBias,ULatentScaler,WPair,WBias,bsxfun(@plus,WBilinear,WBilinear'));
                newScore = testError.auc;
                trainError = convergenceScoreTr(U',UBias,ULatentScaler,WPair,WBias,WBilinear+WBilinear');
                newScoreTr = trainError.auc;                
            else
                testError = convergenceScore(U',UBias,ULatentScaler,WPair,WBias,WBilinear); newScore = testError.auc;
                trainError = convergenceScoreTr(U',UBias,ULatentScaler,WPair,WBias,WBilinear); newScoreTr = trainError.auc;
            end
        else
            newScoreTr = 0; newScore = 0; trainError = []; testError = [];
        end

        % Stop early if parameters blow up/we have many bad successive epochs
        if any(isnan(U(:))) || any(isnan(WBilinear(:))) || isnan(newScore) || (0 && newScore < lastScore + 1e-4)
            U = UOld;
            UBias = UBiasOld;
            WPair = WPairOld;
            WBias = WBiasOld;
            WBilinear = WBilinearOld;

            disp(sprintf('early stopping at epoch %d: auc = %.4g -> %.4g', e, lastScore, newScore));
            break;
        end

        lastScore = newScore;
        UOld = U; UBiasOld = UBias;
        WPairOld = WPair; WBiasOld = WBias;
        WBilinearOld = WBilinear;
    end

    U = U';
    WBilinear = bsxfun(@plus, WBilinear, WBilinear');
