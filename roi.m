function lgraph=roi()
    input = imageInputLayer([256 256 3], 'Name', 'roiInput', 'Normalization', 'none');

    layers=[
       convolution2dLayer(3,64,'Padding','same','Name','ROIconvLayer1')
       reluLayer('Name','ROIreluLayer1')

       convolution2dLayer(3,32,'Padding','same','Name','ROIconvLayer2')
       reluLayer('Name','ROIreluLayer2')

       convolution2dLayer(1,1,'Padding','same','Name','ROIconvLayer3')
       sigmoidLayer('Name','ROIsigmoidLayer') 
        ];

    lgraph = layerGraph(input);
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'roiInput', 'ROIconvLayer1');
end