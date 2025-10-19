clc; close all; clear;

load('trainedFaceModel.mat', 'dlnetEnc', 'dlnetROI', 'dlnetDec');
fprintf('Model loaded\n');

testImage = imread(fullfile( 'face.png'));
testImage = imresize(testImage, [64 64]);
testImage = im2single(testImage);
if size(testImage,3)==1
    testImage = cat(3,testImage,testImage,testImage);
end
dlX = dlarray(testImage,'SSC');

F = predict(dlnetEnc, dlX);
Q = predict(dlnetROI, dlX);

roiMask = gather(extractdata(Q));
roiMask = mat2gray(roiMask); 

F_low  = F(:,:,1:128,:);
F_high = F(:,:,129:end,:);

if exist('rateAllocate', 'file')
    allocated = rateAllocate(F_low, F_high, extractdata(Q));
else
    allocated = F; 
end

reconstructed = predict(dlnetDec, allocated);

origImg = im2uint8(testImage);
reconImg = gather(extractdata(reconstructed));
reconImg = im2uint8(mat2gray(reconImg)); 

mseError = mean((double(origImg(:)) - double(reconImg(:))).^2);
fprintf('Reconstruction MSE: %.4f\n', mseError);

figure('Name','Model Test Results','NumberTitle','off');
subplot(1,3,1);
imshow(origImg);
title('Original Image');

subplot(1,3,2);
imshow(roiMask);
title('Predicted ROI Mask');

subplot(1,3,3);
imshow(reconImg);
title('Reconstructed Image');

fprintf('Test completed\n');
