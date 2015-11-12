function [ precision,recall,map ] = analyze( GTrue,PPred, )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


end


function [precision,recall] = get_precision_recall()
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

