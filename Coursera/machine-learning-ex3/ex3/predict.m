function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

X = [ones(m,1) X]; %add bias unit for traininig set
a_1 = sigmoid(Theta1 * X');

a_1 = a_1';
a_1 = [ones(m,1) a_1]; %add bias unit for training set for hidden layer
a_2 = sigmoid(Theta2 * a_1');

[val,p_temp] = max(a_2,[],1);
p = p_temp';
end
