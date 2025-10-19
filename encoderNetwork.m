function lgraph=encoderNetwork()
    input=imageInputLayer([256 256 3],'Name','input','Normalization','none');

    baseLayers=[
        convolution2dLayer(5,128,'Stride',2,'Padding','same','Name','convLayer1')
        reluLayer('Name','reluLayer1')
        GDN(128,'GDNlayer1')

        convolution2dLayer(5,256,'Stride',2,'Padding','same','Name','convLayer2')
        reluLayer('Name','reluLayer2')
        GDN(256,'GDNlayer2')
        ];
    
    basicQuality=[
        convolution2dLayer(3,128,'Padding','same','Name','convLayer3')
        reluLayer('Name','reluLayer3')
        ];
    
    highQuality=[
        convolution2dLayer(3,320,'Padding','same','Name','convLayer4')
        reluLayer('Name','reluLayer4')
        ];
    concatenationLayer=depthConcatenationLayer(2,'Name','ConcatLayer');
   
    lgraph = layerGraph(input);
    lgraph = addLayers(lgraph, baseLayers);
    lgraph = addLayers(lgraph, basicQuality);
    lgraph = addLayers(lgraph, highQuality);
    lgraph = addLayers(lgraph, concatenationLayer);

    lgraph = connectLayers(lgraph, 'input', 'convLayer1');
    lgraph = connectLayers(lgraph,'GDNlayer2','convLayer3');
    lgraph = connectLayers(lgraph,'GDNlayer2','convLayer4');
    lgraph = connectLayers(lgraph,'reluLayer3','ConcatLayer/in1');
    lgraph = connectLayers(lgraph,'reluLayer4','ConcatLayer/in2');
end