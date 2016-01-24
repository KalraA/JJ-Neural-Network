load Data.txt;
load Times.txt;
oldpred = 10000;
lambdabefore = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100];
hoursbefore = [12*60, 3*60, 2*60, 1.5*60, 1*60, 45, 30, 15];
nodesSize = [1, 2, 3, 5, 10, 20, 40, 80];


for jj = 1:size(hoursbefore, 2)

data = hoursbefore(jj)/5
Data = Data(1:(size(Data, 1) - mod(size(Data, 1), data)));
roundz = size(Data, 1)/data
a = reshape(Data, data, roundz);
b = []
for q = 1:data - 1
b = [b, reshape(Data((q + 1):(size(Data, 1)-data+q)), data, roundz - 1)];
end
d = [a, b]';
%c = 4*roundz/5;

X = b(1:data-1, :)';
yy = b(data, :)';
testData = a(1:data-1, :)';
tt = a(data, :)';

y = zeros(size(yy, 1), 1);
testAns = zeros(size(tt, 1), 1);

for ww = 1:size(y, 1)
	if yy(ww) > 198
		y(ww, 1) = 5;
	elseif yy(ww) > 180
		y(ww, 1) = 4;
	elseif yy(ww) > 70
		y(ww, 1) = 3;
	elseif yy(ww) > 63
		y(ww, 1) = 2;
	else
		y(ww, 1) = 1;
	end
end
max(y)
for rr = 1:size(tt, 1) - 1
	if tt(rr) > 198
		testAns(rr, 1) = 5;
	elseif tt(rr) > 180
		testAns(rr, 1) = 4;
	elseif tt(rr) > 70
		testAns(rr, 1) = 3;
	elseif tt(rr) > 63
		testAns(rr, 1) = 2;
	else
		testAns(rr, 1) = 1;
	end
end
Z = [X, y];
W = randperm(size(Z, 1), size(Z, 1) - floor(size(Z, 1)/1.2));
X = Z(W, 1:data-1);
y = Z(W, data);
m = size(X, 1);



for ii = 1:size(lambdabefore, 2)
for hh = 1:size(nodesSize, 2)
input_layer_size = data - 1;
hidden_layer_size = nodesSize(hh);
num_labels = 5;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = lambdabefore(ii);


%for zz = 1:50
options = optimset('MaxIter', 100);


%W = randperm(size(d, 1), 90);
%X = d(W, 1:data-1);
%y = d(W, data);

costFunction = @(p) gradCostFn(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


billy = cost(end)


%end]

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

if billy < oldpred
	oldpred =  billy;
	bestLambda = lambdabefore(ii);
	bestTime = hoursbefore(ii);
	bestHidden = nodesSize(hh);
end
end
end
end
data = hoursbefore(jj)/5
Data = Data(1:(size(Data, 1) - mod(size(Data, 1), data)));
roundz = size(Data, 1)/data
a = reshape(Data, data, roundz);
b = []
for q = 1:data - 1
b = [b, reshape(Data((q + 1):(size(Data, 1)-data+q)), data, roundz - 1)];
end
d = [a, b]';
%c = 4*roundz/5;

X = b(1:data-1, :)';
yy = b(data, :)';
testData = a(1:data-1, :)';
tt = a(data, :)';

y = zeros(size(yy, 1), 1);
testAns = zeros(size(tt, 1), 1);

for ww = 1:size(y, 1)
	if yy(ww) > 198
		y(ww, 1) = 5;
	elseif yy(ww) > 180
		y(ww, 1) = 4;
	elseif yy(ww) > 70
		y(ww, 1) = 3;
	elseif yy(ww) > 63
		y(ww, 1) = 2;
	else
		y(ww, 1) = 1;
	end
end
max(y)
for rr = 1:size(tt, 1) - 1
	if tt(rr) > 198
		testAns(rr, 1) = 5;
	elseif tt(rr) > 180
		testAns(rr, 1) = 4;
	elseif tt(rr) > 70
		testAns(rr, 1) = 3;
	elseif tt(rr) > 63
		testAns(rr, 1) = 2;
	else
		testAns(rr, 1) = 1;
	end
end

m = size(X, 1);

input_layer_size = data - 1;
hidden_layer_size = bestHidden;
num_labels = 5;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = lambdabefore(jj);

Z = [X, y];

for zz = 1:300
options = optimset('MaxIter', 100);

W = randperm(size(Z, 1), size(Z, 1) - floor(size(Z, 1)/10));
X = Z(W, 1:data-1);
y = Z(W, data);

costFunction = @(p) gradCostFn(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


initial_nn_params = nn_params;


end
X = Z(1:size(Z, 1), 1:data-1);
y = Z(1:size(Z, 1), data);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

size(X);
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


pred = predict(Theta1, Theta2, testData);
billy = mean(double(pred == testAns)) * 100;
fprintf('\n Testing Set Accuracy: %f\n', mean(double(pred == testAns)) * 100);
