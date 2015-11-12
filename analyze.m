function analyze_handler = analyze
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    analyze_handler.get_precison_recall = @get_precision_recall;
    analyze_handler.get_results =  @get_results;

end

function results = get_results(PPred,GTrue,v,DTe,ITe)
    testLinks = sub2ind(size(PPred), DTe(1,:), DTe(2,:));
    X = (v(ITe)+1)/2;
    Y = transpose(PPred(testLinks)) >= 0.5 ;
    [precision,recall] = get_precision_recall(X,Y);
    results.recall = recall;
    results.precision = precision;
end


function [precision,recall] = get_precision_recall(X,Y)
    tp = 0;
    fp = 0;
    tn = 0;
    fn = 0;
    
    [test_count,~] = size(Y);
    for i = 1:test_count
        if X(i,1) == 1 && Y(i,1) == 1
           tp = tp + 1;
        elseif X(i,1) == 1 && Y(i,1) == 0
           fn = fn + 1;
        elseif X(i,1) == 0 && Y(i,1) == 1
           fp = fp + 1;
        elseif X(i,1) == 0 && Y(i,1) == 0
           tn = tn + 1;
       end
    end
    
    recall = tp / (tp + fn);
    precision = tp / (tp + fp);
end

