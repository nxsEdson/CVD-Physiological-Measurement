function signal = get_ROI_signal(img, mask)

[m,n,c] = size(img);

signal = zeros(1,1,c);
for i = 1:c
    tmp = img(:,:,i);
    signal(1,1,i) = mean(double(tmp(mask)));
end

end