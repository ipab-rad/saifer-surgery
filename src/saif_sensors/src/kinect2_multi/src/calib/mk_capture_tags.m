function [tags, ID, T, R, V, msgImage, msgFrame] = mk_capture_tags(kid, datapath, Text)
%% detect tags in ros msg image received from kinect
if ~exist('datapath','var')
    datapath = pwd;
end
%% get image from camera
if kid >= 5
    rosTopic = sprintf('/realsense_%d/color/',kid);
    imgTopic = [rosTopic 'image_raw'];
    %fx = [616.50390625 616.50390625];
    %px = [320.3089599609375 233.7106475830078];
else
    rosTopic = sprintf('/kinect_%d/hd/',kid);
    imgTopic = [rosTopic 'image_color_rect'];
    %fx = [1050.0 1050.0];
    %px = [974.631617845697 539.610033309229];
end
sub = rossubscriber(imgTopic);
fprintf('Waiting for %s msg... ',imgTopic); tic;
pause(1);
msg = receive(sub,10);
msgFrame = msg.Header.FrameId;
msgImage = msg.readImage;
sub.delete();
toc;
imgPath = sprintf('%s/K%d_image_color_rect_hd.png',datapath,kid);
imwrite(msgImage,imgPath);

%% cam info 
infoTopic = [rosTopic 'camera_info'];
sub = rossubscriber(infoTopic);
fprintf('Waiting for %s msg... ',infoTopic); tic;
pause(1);
msgInfo = receive(sub,10);
K = reshape(msgInfo.K,3,3)';
fx = [K(1,1) K(2,2)];
px = [K(1,3) K(2,3)];
sub.delete();
toc;

%% detect tags
tagImage = sprintf('%s/K%d_image_color_rect_hd_tags.png',datapath,kid);
tagPath = sprintf('%s/K%d_image_color_rect_hd_tags.yml',datapath,kid);
%cmd = sprintf('aprilgrid/ethz_apriltag2/build/devel/lib/ethz_apriltag2/apriltags2 -I %s -D %s -S 0.061 > %s', imgPath,tagImage,tagPath);
cmd = sprintf( ...
    '%s -I %s -D %s -S 0.088 -F %f -G %f -P %f -Q %f > %s', ...
    'aprilgrid/ethz_apriltag2/build/devel/lib/ethz_apriltag2/apriltags2', imgPath,tagImage,fx(1), fx(2), px(1), px(2), tagPath);
fprintf('Detecting apriltags in %s ... ',imgPath); tic;
%eval(['!' cmd]);
[status,cmdout] = unix(cmd,'-echo');
tags = YAML.read(tagPath); toc;
if isempty(tags)
    error(['mk_capture_tags:cannot run aprilgrid [\n' cmdout]);
end
%% parse tags
ids = fieldnames(tags);
tagCount = length(ids);
% translation vector from camera to the April tag
T = zeros(tagCount,3);
% orientation of April tag with respect to camera: the camera
% convention makes more sense here, because yaw,pitch,roll then
% naturally agree with the orientation of the object
R = zeros(tagCount,3);
% orientation vector
V = zeros(tagCount,3);
ID = zeros(tagCount,1);
for i = 1:tagCount
    %% parse
    ti = ids{i};
    ID(i) = str2num(ti(3:end));
    Th = [tags.(ti).translation 1]*Text.T;
    tags.(ti).translation = Th(1:3);
    T(i,:) = tags.(ti).translation;
    R(i,:) = tags.(ti).rotation;
    
    Rotm = eul2rotm(R(i,:));
    V(i,:) = Rotm(:,1);
end

