%% Select the frames.
% To select the important frames uncomment next line
% [Images, TotalFrames] = selectKeyFramesFromBag(Path)
% All the frames -> FramesToProcess = 0;

%FramesToProcess = [1,19,32,40,47,55,63,72,82,89,97,106,112,123,138,148,155,164,171,178,186,195,202,210,219,227,234,243,254]; %K1
FramesToProcess = [1,19,32,40,47,63,72,82,89,97,106,115,123,155,164,171,178,186,195,219,227,234,243,254]; %K1 K2 K3 K4

%% get the points, color image and depth image from the bag file
%[points, rgb, depth] = getDataFromBag('/Users/Marcelo/GDriveUA/Calibration/2018-07-17-17-39-34.bag', FramesToProcess);
[points, rgb, depth] = getDataFromBag('/home/ratchet/Bishop/MultiKinect/calib/ball/2018-07-24-20-07-08.bag', FramesToProcess);

%% segment the data based on color information
[SegmentedPoints, SegmentedRGB, SegmentedDepth, Indices] = segmentationBallCalibration(points, rgb, depth);

%% estimate the center of the sphere
[CentreSpheres, DTotal, StdTotal] = estimateCentreSphere(SegmentedPoints);

%% load T1 from aprilgrid
addpath('yaml');

datapath = [pwd '/aprilgrid/data/april5'];
tags{1} = YAML.read(sprintf('%s/K1_image_color_rect_hd_tags.yml',datapath));
tags{2} = YAML.read(sprintf('%s/K2_image_color_rect_hd_tags.yml',datapath));
tags{3} = YAML.read(sprintf('%s/K3_image_color_rect_hd_tags.yml',datapath));
tags{4} = YAML.read(sprintf('%s/K4_image_color_rect_hd_tags.yml',datapath));

TR_tags = mk_transform_tags(tags);

%% find the transformation
TR = mk_transform_ball(CentreSpheres,TR_tags{1}.T);

% datapath = [pwd '/ball'];
% launchpath = [datapath '/tf_reg_K1234.launch'];
% mk_export_calib_ros(TR, launchpath);
% fprintf('Extrinsic calibration exported to %s.\n',launchpath);


%% apply transformation

[CentreSphereCorrected,PointsCorrected] = mk_applyTransform_calib(CentreSpheres,points,rgb,TR);
[CentreSphereCorrected_tags,PointsCorrected_tags] = mk_applyTransform_calib(CentreSpheres,points,rgb,TR_tags);

%% Calculate variance of corresponding centres
clear CentreMatrix
[CentreMatrix, CentreMean, DKinect] = mk_calculate_variance(CentreSphereCorrected);
[CentreMatrix_tags, CentreMean_tags, DKinect_tags] = mk_calculate_variance(CentreSphereCorrected_tags);

figure; plot(DKinect');
figure; plot(DKinect_tags');

%% Plotting

mk_plot_centres(CentreMatrix, CentreMean, PointsCorrected);
mk_plot_centres(CentreMatrix_tags, CentreMean_tags, PointsCorrected_tags);

%% Create Ply
PntAux = PointsCorrected{1}.Location;
PntAux = cat(1,PntAux, PointsCorrected{2}.Location, PointsCorrected{3}.Location, PointsCorrected{4}.Location);
PntCAux = PointsCorrected{1}.Color;
PntCAux = cat(1,PntCAux, PointsCorrected{2}.Color, PointsCorrected{3}.Color, PointsCorrected{4}.Color);

pcwrite(pointCloud(PntAux,'Color',PntCAux),'PointsCorrected.ply','Encoding','ascii');

PntAux_tags = PointsCorrected_tags{1}.Location;
PntAux_tags = cat(1,PntAux_tags, PointsCorrected_tags{2}.Location, PointsCorrected_tags{3}.Location, PointsCorrected_tags{4}.Location);
PntCAux_tags = PointsCorrected_tags{1}.Color;
PntCAux_tags = cat(1,PntCAux_tags, PointsCorrected_tags{2}.Color, PointsCorrected_tags{3}.Color, PointsCorrected_tags{4}.Color);
pcwrite(pointCloud(PntAux_tags,'Color',PntCAux_tags),'PointsCorrected_tags.ply','Encoding','ascii');


