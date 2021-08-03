
function cnn = TrainingCNNClassifier(train_x, train_y, ChangeTargets, no_of_epochs, batch_size,h, w)

%% Determining layers
if max(unique(ChangeTargets))==2
    no_of_feature_maps = 2;
    OutClass = 2;
else
    no_of_feature_maps = max(unique(ChangeTargets));
    OutClass = max(unique(ChangeTargets));
end

cnn.namaste=1; 
cnn=initcnn(cnn,[h w]);
cnn=cnnAddConvLayer(cnn, no_of_feature_maps, [batch_size batch_size], 'rect');
cnn=cnnAddPoolLayer(cnn, 1, 'mean');
cnn=cnnAddConvLayer(cnn, 100, [1 1], 'tanh');
cnn=cnnAddPoolLayer(cnn, 1, 'mean');
cnn=cnnAddFCLayer(cnn,200, 'tanh' ); 
cnn=cnnAddFCLayer(cnn,OutClass, 'sigm' ); 

%% Dispalying results
display 'training started...Please wait for few minutes...'
tic
cnn=traincnn(cnn,train_x,train_y, no_of_epochs,batch_size);
toc
display '...training finished.'

tic
end