classdef GDN < nnet.layer.Layer
    properties (Learnable)   
        beta  
        gamma  
    end
    methods
        function layer=GDN(numOfChannels,nameOfLayer)
            layer.Name=nameOfLayer;
            layer.beta = rand(numOfChannels,1,'single');   
            layer.gamma = eye(numOfChannels,'single');   
        end

        function output=predict(layer,X) 
            [H,W,C,N] = size(X);
            beta=abs(layer.beta);
            gamma=abs(layer.gamma);
            X_reshaped = reshape(X, H*W, C, N); % H*W x C x N
    Z_reshaped = zeros(size(X_reshaped), 'like', X);

    for n = 1:N
        for i = 1:C
            norm_i = sqrt(beta(i) + sum(gamma(i,:) .* squeeze(X_reshaped(:, :, n)).^2, 2));
            Z_reshaped(:, i, n) = squeeze(X_reshaped(:, i, n)) ./ norm_i;
        end
    end

    output = reshape(Z_reshaped, H, W, C, N);
        end
    end


end
