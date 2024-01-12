%%%%% Matlab essentials for handling digit images and training a simple
%%%%% linear classifier
%%%%%
%%%%% Herbert Jaeger, Nov 18, 2016
%%%%%
%%%%% This script loads the raw image file into Matlab, displays some digit
%%%%% images, and trains (by linear regression) a classifier that uses 10
%%%%% features extracted from the images. These features are derived from
%%%%% ten "prototype" images p_i obtained by averaging all images of one class;
%%%%% the i-th feature value extracted from an image vector x is the
%%%%% product p_i' * x. (the p_i are named "meanTrainImages" in the code below).  


% load the pixel data, resulting in a matlab matrix of dim 2000 x 240
% called "mfeat_pix"
load mfeat-pix.txt -ascii;

% plot the figure from the lecture notes. 
figure(1);
for i = 1:10
    for j = 1:10
        pic = mfeat_pix(200 * (i-1)+j ,:);  
        picmatreverse = zeros(15,16);
        % the filling of (:) is done columnwise!
        picmatreverse(:)= - pic;
        picmat = zeros(15,16);
        for k = 1:15
            picmat(:,k)=picmatreverse(:,16-k);
        end
        subplot(10,10,(i-1)* 10 + j);
        pcolor(picmat');
        axis off;
        colormap(gray(10));
    end
end

% split the data into a training and a testing dataset
pickIndices = [1:100 201:300 401:500 601:700 801:900 ...
    1001:1100 1201:1300 1401:1500 1601:1700 1801:1900];
trainPatterns = mfeat_pix(pickIndices,:);
testPatterns = mfeat_pix(pickIndices + 100, :);

% create indicator matrices size 10 x 1000 with the class labels coded by 
% binary indicator vectors
b = ones(1,100);
trainLabels = blkdiag(b, b, b, b, b, b, b, b, b, b);
testLabels = trainLabels;

% create a row vector of correct class labels (from 1 ... 10 for the 10
% classes). This vector is the same for training and testing.
correctLabels = [b 2*b 3*b 4*b 5*b 6*b 7*b 8*b 9*b 10*b];

%%%%% from here, a demo implementation of a linear classifer based on 
%%%%% the ten class-mean features (hand-made features f_3 from the 
%%%%% lecture notes)

meanTrainImages = zeros(240, 10);
for i = 1:10
    meanTrainImages(:,i) = mean(trainPatterns((i-1)*100+1:i*100, :))';
end

featureValuesTrain = meanTrainImages' * trainPatterns';
featureValuesTest = meanTrainImages' * testPatterns';

% compute linear regression weights W
W = (inv(featureValuesTrain * featureValuesTrain') * ...
    featureValuesTrain * trainLabels')';

% compute train misclassification rate
classificationHypothesesTrain = W * featureValuesTrain;
[maxValues maxIndicesTrain] = max(classificationHypothesesTrain);
nrOfMisclassificationsTrain = sum(correctLabels ~= maxIndicesTrain);
disp(sprintf('train misclassification rate = %0.3g', ...
    nrOfMisclassificationsTrain / 1000));

% compute test misclassification rate
classificationHypothesesTest = W * featureValuesTest;
[maxValues maxIndicesTest] = max(classificationHypothesesTest);
nrOfMisclassificationsTest = sum(correctLabels ~= maxIndicesTest);
disp(sprintf('test misclassification rate = %0.3g', ...
    nrOfMisclassificationsTest / 1000));

%%%%%%%%%% now with basic ridge regression on raw pics
%%
alpha = 10000;
Wridge = (inv(trainPatterns' * trainPatterns + alpha * eye(240)) * ...
    trainPatterns' * trainLabels')';

% compute train misclassification rate
classificationHypothesesTrain_ridge = Wridge * trainPatterns';
[maxValues maxIndicesTrain] = max(classificationHypothesesTrain_ridge);
nrOfMisclassificationsTrain = sum(correctLabels ~= maxIndicesTrain);
disp(sprintf('train misclassification rate ridge = %0.3g', ...
    nrOfMisclassificationsTrain / 1000));

% compute test misclassification rate
classificationHypothesesTest_ridge = Wridge * testPatterns';
[maxValues maxIndicesTest] = max(classificationHypothesesTest_ridge);
nrOfMisclassificationsTest = sum(correctLabels ~= maxIndicesTest);
disp(sprintf('test misclassification rate ridge = %0.3g', ...
    nrOfMisclassificationsTest / 1000));


% Note: it is instructive to display images of misclassified digits
% together with their correct and estimated class labels - not done here









