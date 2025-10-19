function lgraph =decoderNetwork()
    input = imageInputLayer([64 64 448],'Name','decoderInput','Normalization','none'); 

    layers = [
        transposedConv2dLayer(5,256,'Stride',2,'Cropping','same','Name','decoderConv1')
        reluLayer('Name','relu1')
        GDN(256,'IGDN1')
        
        transposedConv2dLayer(5,128,'Stride',2,'Cropping','same','Name','decoderConv2')
        reluLayer('Name','relu2')
        GDN(128,'IGDN2')
        
        convolution2dLayer(3,3,'Padding','same','Name','reconstruction')
        sigmoidLayer('Name','decoderSigmoidLayer')
        ];
    
    lgraph = layerGraph(input);
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'decoderInput', 'decoderConv1');
end