function g = sigmoidGradient(z)

g = zeros(size(z));


a = sigmoid(z);
g = a.*(1 - a);



end
