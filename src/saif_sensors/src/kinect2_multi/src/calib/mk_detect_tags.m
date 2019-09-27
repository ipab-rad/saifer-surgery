function [ID, T, R, V] = mk_detect_tags(kid)

%% read file
addpath '../yaml';

%img = imread('data/K1_color_rect.png');
%img = rgb2gray(img);
%tags = apriltags2(img,'36h19');

tags = YAML.read(sprintf('data/K%d_color_rect.yaml',kid));
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
    %%
    ti = ids{i};
    ID(i) = str2num(ti(3:end));
    T(i,:) = tags.(ti).translation;
    R(i,:) = tags.(ti).rotation;
    
    Rotm = eul2rotm(R(i,:));
    V(i,:) = diag(Rotm);
end

