if ~exist('calibrationTrans.mat','file')
    calculateTransforms
else
    load 'calibrationTrans.mat'
end
% If there is transformation
if exist('TR','var')
    
    % read the point clouds
    [K1_points,K1_color,~,~]=read_ply('K1_hd_pcl_rgb.ply');
    [K2_points,K2_color,~,~]=read_ply('K2_hd_pcl_rgb.ply');
    [K3_points,K3_color,~,~]=read_ply('K3_hd_pcl_rgb.ply');
    [K4_points,K4_color,~,~]=read_ply('K4_hd_pcl_rgb.ply');
    
    
    K1_rgb = imread('K1_hd_image_color_rect.png');
    K1_depth = imread('K1_hd_image_depth_rect.png');
    K1_hsv = rgb2hsv(K1_rgb);
    K1_grey = rgb2gray(K1_rgb);

    K2_rgb = imread('K2_hd_image_color_rect.png');
    K2_depth = imread('K2_hd_image_depth_rect.png');
    K2_hsv = rgb2hsv(K2_rgb);
    K2_grey = rgb2gray(K2_rgb);

    K3_rgb = imread('K3_hd_image_color_rect.png');
    K3_depth = imread('K3_hd_image_depth_rect.png');
    K3_hsv = rgb2hsv(K3_rgb);
    K3_grey = rgb2gray(K3_rgb);

    K4_rgb = imread('K4_hd_image_color_rect.png');
    K4_depth = imread('K4_hd_image_depth_rect.png');
    K4_hsv = rgb2hsv(K4_rgb);
    K4_grey = rgb2gray(K4_rgb);


    [ K1_grey_cut, K1_depth_cut, K1_points_cut, K1_ind_remove, K1_mask ] = segmentDistance( K1_grey, K1_depth, K1_points, 2500 );
    [ K2_grey_cut, K2_depth_cut, K2_points_cut, K2_ind_remove, K2_mask ] = segmentDistance( K2_grey, K2_depth, K2_points, 2500 );
    [ K3_grey_cut, K3_depth_cut, K3_points_cut, K3_ind_remove, K3_mask ] = segmentDistance( K3_grey, K3_depth, K3_points, 2500 );
    [ K4_grey_cut, K4_depth_cut, K4_points_cut, K4_ind_remove, K4_mask ] = segmentDistance( K4_grey, K4_depth, K4_points, 2500 );
    
  	K1_color_cut = K1_color(K1_ind_remove,:);
	K2_color_cut = K2_color(K2_ind_remove,:);
	K3_color_cut = K3_color(K3_ind_remove,:);
 	K4_color_cut = K4_color(K4_ind_remove,:);
    
    % transform the views with the calibration transformations
    K2_points_transform = transformPcl(K2_points_cut,TR{2}.T);
    K3_points_transform = transformPcl(K3_points_cut,TR{3}.T);
    K4_points_transform = transformPcl(K4_points_cut,TR{4}.T);
 
	Fusion = cat(1,K1_points_cut,K2_points_transform,K3_points_transform,K4_points_transform);
    FusionColor = cat(1,K1_color_cut,K2_color_cut,K3_color_cut,K4_color_cut);

    if 0
    figure; pcshow(Fusion,FusionColor./255)
%         scatter3(K1_points_cut(:,1),K1_points_cut(:,2),K1_points_cut(:,3),1,K1_color_cut./255);axis image;hold on
%         scatter3(K2_points_transform(:,1),K2_points_transform(:,2),K2_points_transform(:,3),1,K2_color_cut./255)
%         scatter3(K3_points_transform(:,1),K3_points_transform(:,2),K3_points_transform(:,3),1,K3_color_cut./255)
%         scatter3(K4_points_transform(:,1),K4_points_transform(:,2),K4_points_transform(:,3),1,K4_color_cut./255)
    end
        
    %% ICP
%         multiViewIcp
end

