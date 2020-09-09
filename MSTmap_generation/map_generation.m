%% MSTmap generation
% This is an example based on the SeetaFace Enginee and VIPL-HR database.
% You may need to adjust the code for your data.
% copyright @ Xuesong Niu
clear all;
addpath('./utils');

%% parameter setting
% location of the face video
video_file = '../data_example/video.avi';

% If you use Seetaface(https://github.com/seetaface/SeetaFaceEngine), you will get 81 facial landmarks; 
% While when you use the OpenFace SDK (https://github.com/TadasBaltrusaitis/OpenFace), you will get 68
% facial landmarks. The landmark_num should be set based on the facial
% landmark engine you use.
landmark_num = 81; 
% location for the landmarks
landmark_dir = '../data_example/face_landmarks_81p/';

% get the ground truth HR for the whole video
gt_file = '../data_example/gt_HR.csv';
gt = xlsread(gt_file);

% get the frame rate for the input face video 
time = load('../data_example/time.txt');
fps = length(time)/(time(end) - time(1))*1000;

% get the BVP signal for the input face video 
BVP_file = '../data_example/wave.csv';
BVP_whole_video = xlsread(BVP_file);

% the video clip length
clip_length = 300;

% location of the generated MSTmaps
Target_path = './MSTmaps/';


%% The MSTmap generation for the whole video
obj = VideoReader(video_file);
numFrames = obj.NumberOfFrames; 

MSTmap_whole_video = zeros(63, numFrames, 6);
for k = 1 : numFrames 
    frame = read(obj,k);
    lmk_path = strcat(landmark_dir, 'landmarks', num2str(k), '.dat');
    %% processing for each frame
    if(exist(lmk_path))
        fid = fopen(lmk_path, 'r');
        if fid > 0
            landmarks = fread(fid,inf,'int');
            landmarks = reshape(landmarks, [2, landmark_num]);
        end
        fclose(fid);

        % get the landmarks for each video frame.
        % landmarks are sorted in a 2*landmark_num martrix in the format of 
        % [x0 x1 x2 ... xn; y0 y1 y2 ... yn];
        MSTmap_whole_video = GenerateSignalMap(MSTmap_whole_video, frame, k, landmarks, landmark_num);
    end
end

%% Save MSTmaps for the video clips
idx = 1;
% Since the frame rate of VIPL-HR is not stable, we use the number of the 
% heart beats as the ground truth of the video clip for normalization. 
% i.e., heart_beats_num = gt_HR * clip_length / fps / 60;
idx = save_MSTmaps(Target_path, MSTmap_whole_video, BVP_whole_video, gt, fps, clip_length, idx);