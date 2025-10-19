clc; close all;

rootFolder = 'faces/faces';
imds = imageDatastore(rootFolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions',{'.png','.jpg'}, ...
    'LabelSource','foldernames');

maxImages = min(50, numel(imds.Files));
imds.Files = imds.Files(1:maxImages);
fprintf('Loaded %d images for quick training.\n', numel(imds.Files));


inputSize = [64 64 3];    
miniBatchSize = 2;
imds.ReadFcn = @(filename) imresize(imread(filename), [inputSize(1) inputSize(2)]);

augimds = augmentedImageDatastore(inputSize, imds);

dlnetEnc = dlnetwork(encoderNetwork());
dlnetROI = dlnetwork(roi());
dlnetDec = dlnetwork(decoderNetwork());

numEpochs = 50;
learnRate = 1e-3;

mbq = minibatchqueue(augimds, ...   
    'MiniBatchSize', miniBatchSize, ...
    'MiniBatchFcn', @(x) preprocessBatch(x), ...
    'MiniBatchFormat','SSCB');

avgGradEnc = []; avgSqGradEnc = [];
avgGradROI = []; avgSqGradROI = [];
avgGradDec = []; avgSqGradDec = [];

for epoch = 1:numEpochs
    reset(mbq);
    iteration = 0;
    while hasdata(mbq)
        iteration = iteration + 1;
        dlX = next(mbq);
    
        fprintf('Epoch %d | Iter %d ... ', epoch, iteration);
        drawnow limitrate nocallbacks;  
    
        [loss, gradEnc, gradROI, gradDec] = dlfeval(@modelGradients, dlnetEnc, dlnetROI, dlnetDec, dlX);

        [dlnetEnc, avgGradEnc, avgSqGradEnc] = adamupdate(dlnetEnc, gradEnc, avgGradEnc, avgSqGradEnc, iteration, learnRate);
        [dlnetROI, avgGradROI, avgSqGradROI] = adamupdate(dlnetROI, gradROI, avgGradROI, avgSqGradROI, iteration, learnRate);
        [dlnetDec, avgGradDec, avgSqGradDec] = adamupdate(dlnetDec, gradDec, avgGradDec, avgSqGradDec, iteration, learnRate);

        if mod(iteration,5)==0
            fprintf('Epoch %d | Iter %d | Loss = %.4f\n', epoch, iteration, double(gather(extractdata(loss))));
        end
    end
end

fprintf('Training complete.\n');

save('trainedFaceModel.mat','dlnetEnc','dlnetROI','dlnetDec');
fprintf('Model saved as trainedFaceModel.mat\n');

function dlX = preprocessBatch(X)
    X = cat(4,X{:});
    X = im2single(X);
    dlX = dlarray(X,'SSCB');
end

function [loss,gradEnc,gradROI,gradDec] = modelGradients(dlnetEnc,dlnetROI,dlnetDec,dlX)
    F = forward(dlnetEnc, dlX);
    Q = forward(dlnetROI, dlX);
    
    F_low  = F(:,:,1:128,:);
    F_high = F(:,:,129:end,:);
    
    allocated = rateAllocate(F_low,F_high,extractdata(Q));
    reconstructed = forward(dlnetDec,allocated);
    
    loss = mean((reconstructed - dlX).^2,'all');
    
    [gradEnc, gradROI, gradDec] = dlgradient(loss, dlnetEnc.Learnables, dlnetROI.Learnables, dlnetDec.Learnables);
end
