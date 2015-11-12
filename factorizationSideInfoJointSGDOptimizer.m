% Stochastic gradient optimization of latent features + side-information
%
% D = 3 x pairs, where each column = (node id, node id, edge status), edge status = {0, 1}
% sidePair = dPair x |V| x |V|, |V| = # of nodes; features for each pair of nodes
% sideBilinear = dBilinear x |V|; features for each node
% weights = structure with all the learned weights inside it
% lambda = structure with all the regularization parameters inside it
% eta = structure with all the learning rates inside it
% EPOCHS = # of sweeps over D
% epochFrac = used if we only want to do a partial sweep over D
% convergenceScore{Tr,Te} = function that computes error on train/test set
% loss = {'square', 'logistic'}
% link = {'none', 'sigmoid'}
% symmetric = {0, 1}; is the graph symmetric?
% covariance = precomputed values of sideBilinear(:,i)*sideBilinear(:,j)'
function [U, UBias, ULatentScaler, WPair, WBias, WBilinear, trainError, testError] = factorizationSideInfoJointSGDOptimizer(D, sidePair, sideBilinear, weights, lambda, eta, EPOCHS, epochFrac, convergenceScoreTr, convergenceScore, loss, link, symmetric, covariance, varargin)

    pairs = size(D, 2);
    %disp(sprintf('processing %d pairs (1e%d)', pairs, round(log10(pairs))));

    square = strcmp(loss,'square');
    squareHinge = strcmp(loss,'squareHinge');
    sigmoid = strcmp(link,'sigmoid');

    % Extracting weights from the appropriate structures
    U0 = weights.U; U0Bias = weights.UBias; ULatentScaler = weights.ULatentScaler;
    WPair = weights.WPair; WBias = weights.WBias; WBilinear = weights.WBilinear;
    
    if ~isstruct(lambda)
        lambdaLatent = lambda; lambdaRowBias = lambda; lambdaLatentScaler = lambda; lambdaPair = lambda; lambdaBilinear = lambda; lambdaScaler = ones(size(U0Bias));
    else
        lambdaLatent = lambda.lambdaLatent; lambdaRowBias = lambda.lambdaRowBias; lambdaLatentScaler = lambda.lambdaLatentScaler; lambdaPair = lambda.lambdaPair; lambdaBilinear = lambda.lambdaBilinear; lambdaScaler = lambda.lambdaScaler;
    end
    
    if numel(lambdaScaler) == 1
        lambdaScaler = lambdaScaler * ones(size(U0Bias));
    end
    
    if ~isstruct(eta)
        etaLatent0 = eta; etaLatentScaler0 = eta; etaPair0 = eta; etaBilinear0 = eta; etaRowBias0 = eta; etaBias0 = eta;
    else
        etaLatent0 = eta.etaLatent; etaLatentScaler0 = eta.etaLatentScaler; etaPair0 = eta.etaPair; etaBilinear0 = eta.etaBilinear; etaRowBias0 = eta.etaRowBias; etaBias0 = eta.etaBias;
    end

    lowRank = (size(WBilinear,1) ~= size(WBilinear,2));
    cachedCovariance = (numel(covariance{:}) > 0);
    hasDyadicSideInfo = numel(sidePair) > 0;

    U = U0; UBias = U0Bias;    
    UOld = U0; UBiasOld = U0Bias; ULatentScalerOld = ULatentScaler;
    WPairOld = WPair; WBiasOld = WBias; WBilinearOld = WBilinear;

    lastScore = 0; bestScore = 0; badEpochs = 0;
    trainError = []; testError = [];

    %obj = computeObjective(U,UBias,ULatentScaler,WBilinear,WPair,WBias,sideBilinear,sidePair,D,square,sigmoid);
    %disp(sprintf('initial objective: %.8g', obj));

    %% Main SGD body
    for e = 1 : EPOCHS
        % Dampening of the learning rates across epochs
        etaLatent = etaLatent0/((1 + etaLatent0*lambdaLatent)*e);
        etaRowBias = etaRowBias0/((1 + etaRowBias0*lambdaRowBias)*e);
        etaLatentScaler = etaLatentScaler0/((1 + etaLatentScaler0*lambdaLatentScaler)*e);
        etaPair = etaPair0/((1 + etaPair0*lambdaPair)*e);
        etaBilinear = etaBilinear0/((1 + etaBilinear0*lambdaBilinear)*e);        
        etaBias = etaBias0/((1 + etaBias0*lambdaLatent)*e);

        % Random shuffle of the training set
        I = randperm(pairs);
        D = D(:,I);

        for t = 1 : round(epochFrac * pairs)
            %examples = (t-1)*1+1:min(pairs,t); % for varying batch size
            examples = t;

            i = D(1,examples);
            j = D(2,examples);
            truth = D(3,examples)';
            
            %% Prediction
            prediction = (U(:,i)' * ULatentScaler * U(:,j) + UBias(i) + UBias(j))'; % E x 1
            
            if hasDyadicSideInfo
                prediction = prediction + WPair * sidePair(:,i,j) + WBias;
            end

            % Only update (potentially expensive) bilinear component when
            % required, viz. learning rate is non-zero
            if etaBilinear > 0
                if lowRank
                    % Much more efficient to multiply two components separately
                    % and then multiply the results, rather than forming
                    % W'W first and reducing to standard case...
                    prediction = prediction + (WBilinear * sideBilinear(:,i))' * (WBilinear * sideBilinear(:,j));
                else
                    prediction = prediction + sideBilinear(:,i)' * WBilinear * sideBilinear(:,j);
                end
            end

            if sigmoid % Link function
                prediction = 1./(1 + exp(-prediction));
            end

            %% Gradients

            % Common gradient scaler
            gradScaler = (prediction - truth);
            if square
                gradScaler = 2 * gradScaler;
                if sigmoid
                    gradScaler = gradScaler * prediction * (1 - prediction);
                end
            elseif squareHinge
                % TODO
            end

            gradI = ULatentScaler*U(:,j);
            gradJ = ULatentScaler'*U(:,i);
            gradRowBias = ones(1,numel(examples)); % 1 x 1
            gradBias = ones(1,numel(examples));
            if hasDyadicSideInfo,  gradPair = sidePair(:,i,j)'; end;
            
            if etaLatentScaler > 0
                if symmetric
                    gradLatentScaler = diag(U(:,i).*U(:,j));
                else
                    gradLatentScaler = U(:,i)*U(:,j)';
                end
            end

            if etaBilinear > 0
                if lowRank
                    % Again, more efficient to do one inner multiplication first
                    gradBilinear = (WBilinear * sideBilinear(:,i))*sideBilinear(:,j)' + (WBilinear * sideBilinear(:,j))*sideBilinear(:,i)';
                else
                    % Check if d x d matrix has already been computed
                    if cachedCovariance
                        gradBilinear = covariance{i,j};
                    else
                        gradBilinear = sideBilinear(:,i)*sideBilinear(:,j)';
                    end
                end
            end

            % If relationship is symmetric, then update not only for (i,j) but for (j,i) also
            if symmetric
                % Actually, I think only the bilinear component need be
                % updated here, although updating the rest doesn't hurt...
                gradI = gradI + ULatentScaler'*U(:,j);
                gradJ = gradJ + ULatentScaler*U(:,i);
                gradRowBias = gradRowBias + gradRowBias;
                gradBias = gradBias + gradBias;
                if hasDyadicSideInfo, gradPair = gradPair + sidePair(:,j,i)'; end;

                if etaLatentScaler > 0, gradLatentScaler = gradLatentScaler + gradLatentScaler'; end;
                if etaBilinear > 0, gradBilinear = gradBilinear + gradBilinear'; end;
            end

            U(:,[i j]) = U(:,[i j]) - etaLatent * (gradScaler * [gradI gradJ] + lambdaLatent * [lambdaScaler(i)*U(:,i) lambdaScaler(j)*U(:,j)]);            
            UBias([i j]) = UBias([i j]) - etaRowBias * (gradScaler * gradRowBias + lambdaRowBias * UBias([i j]));
            WBias = WBias - etaBias * gradScaler * gradBias;
            if hasDyadicSideInfo, WPair = WPair - etaPair * (gradScaler * gradPair + lambdaPair * WPair); end;

            if etaLatentScaler > 0
                ULatentScaler = ULatentScaler - etaLatentScaler * (gradScaler * gradLatentScaler + lambdaLatentScaler * ULatentScaler);
            end
            
            if etaBilinear > 0
                WBilinear = WBilinear - etaBilinear * (gradScaler * gradBilinear + lambdaBilinear * WBilinear);
            end
        end

        % Periodic information about model performance
        if numel(convergenceScoreTr) > 0 && mod(e,5) == 0
            fprintf('epoch %d ', e);
            trainError = convergenceScoreTr(U',UBias,ULatentScaler,WPair,WBias,WBilinear); newScoreTr = trainError.auc;
            testError = convergenceScore(U',UBias,ULatentScaler,WPair,WBias,WBilinear); newScore = testError.auc;
            fprintf('test auc: %.4g, train auc: %.4g; ', newScore, newScoreTr);

            %obj = computeObjective(U,UBias,WBilinear,WPair,WBias,sideBilinear,sidePair,D,square,sigmoid);
            paramDelta = mean(mean((U - UOld).^2));
            fprintf('objective: %.8g, paramDelta = %.4g\n', obj, paramDelta);            
        else
            newScoreTr = 0; newScore = 0; trainError = []; testError = [];
        end

        % Keeps track of how many epochs in a row have led to limited improvement
        if newScore < lastScore + 1e-4
            badEpochs = badEpochs + 1;
        else
            badEpochs = 0;
        end

        % Stop early if parameters blow up/we have many bad successive epochs
        if any(isnan(U(:))) || any(isnan(WBilinear(:))) || isnan(newScore) || (0 && badEpochs > 3)
            U = UOld;
            UBias = UBiasOld;
            ULatentScaler = ULatentScalerOld;
            WPair = WPairOld;
            WBias = WBiasOld;
            WBilinear = WBilinearOld;

            fprintf('early stopping at epoch %d: auc = %.4g -> %.4g\n', e, lastScore, newScore);
            break;
        end        

        if newScore > bestScore
            UOld = U; UBiasOld = UBias; ULatentScalerOld = ULatentScaler;
            WPairOld = WPair; WBiasOld = WBias;
            WBilinearOld = WBilinear;            

            bestScore = newScore;
        end        

        lastScore = newScore;
    end

    U = U';
