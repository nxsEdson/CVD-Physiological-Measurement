function SignalMapOut = getCombinedSignalMap(SignalMap, ROInum)

All_idx = ff2n(size(ROInum,1));

SignalMapOut = zeros(size(All_idx,1)-1, 1, size(SignalMap,2));
% SignalMapStd = zeros(size(All_idx,1)-1, size(SignalMap,2), size(SignalMap,3));

for i = 2:size(All_idx,1)
    tmp_idx = find(All_idx(i,:)==1);
    
    tmp_signal = SignalMap(tmp_idx, :);
    tmp_ROI = ROInum(tmp_idx);
    tmp_ROI = tmp_ROI./sum(tmp_ROI);
    tmp_ROI = repmat(tmp_ROI, [1,6]);
    
    SignalMapOut(i-1,:,:) = sum(tmp_signal.*tmp_ROI,1);  
end

% Map = SignalMap;
% [m,n,channel] = size(Map);
% std_window = 20;

% for c = 1:channel
%     for i = 1:m
%         sig_temp = Map(i,:,c);
%         SignalMapStd(i,:,c) = movstd(sig_temp, std_window);
%     end
% end

end