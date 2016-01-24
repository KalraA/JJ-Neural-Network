function [J grad] = gradCostFn(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(rows(X), 1), X];
bob = zeros(m, num_labels);
for qq = 1:m
	bob(qq, y(qq)) = 1;
end
%FORWARD PROPAGATION
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
%b is a 5000x10 matrix containing all of parameters.
%Computing the cost function:
t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);

J = (sum(sum(-bob .* log(a3) - (1-bob).*(log(1-a3)))))/m + (lambda/(2*m))*(sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));


%Initialize Big Delta
bigDelta1 = zeros(size(Theta1));
bigDelta2 = zeros(size(Theta2));

%Looping through each example
delta3 = (a3 - bob);
%Compute delta2 or the hidden layer delta
delta2 = delta3*Theta2(:, 2:end) .* sigmoidGradient(z2);
%Figure out big delta:
bigDelta2 = delta3'*a2;
bigDelta1 = delta2'*a1;



Theta1_grad = bigDelta1 / m + [zeros(size(Theta1, 1), 1), (lambda/m).*Theta1(:, 2:end)];
Theta2_grad = bigDelta2 / m + [zeros(size(Theta2, 1), 1), (lambda/m).*Theta2(:, 2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
