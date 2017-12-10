function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
prediction = X * theta;
J =  (1/(2*m))*(sum((prediction.-y).^2)) + ((lambda/(2*m)) * (sum(theta.^2)-theta(1)^2));

theta(1) = 0;
grad = ((1/m) * (X' * (prediction-y))) + ((lambda/m) * theta);

J = J(:);
grad = grad(:);

end
