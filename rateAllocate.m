function allocated = rateAllocate(Fb, Fh, Q)
    if isa(Q, 'dlarray')
        Q = extractdata(Q);
    end

    [H, W, ~] = size(Fb);
    resizedQ = imresize(Q, [H, W]);

    if ndims(resizedQ) == 2
        resizedQ = reshape(resizedQ, [H, W, 1]);
    end

    Q_low  = repmat(resizedQ, [1 1 size(Fb,3)]);
    Q_high = repmat(resizedQ, [1 1 size(Fh,3)]);

    Fb_weighted = Fb .* (1 - Q_low);
    Fh_weighted = Fh .* Q_high;

    allocated = cat(3, Fb_weighted, Fh_weighted);
end
