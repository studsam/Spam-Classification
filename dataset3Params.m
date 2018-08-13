function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
       model = svmTrain(X,y,C,@(x1,x2)gaussianKernel(x1,x2,sigma));
       Predictions = svmPredict(model,Xval);
       initial_prediction_error = mean(double(Predictions~=yval));

   for C_test = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
     for sigma_test = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6]
       model = svmTrain(X,y,C_test,@(x1,x2)gaussianKernel(x1,x2,sigma_test));
       Predictions = svmPredict(model,Xval);
       
       prediction_error = mean(double(Predictions~=yval));
       if(prediction_error < initial_prediction_error)
            initial_prediction_error = prediction_error;
            C = C_test;
            sigma = sigma_test;
       end
   end     
            
   
   


% =========================================================================

end
