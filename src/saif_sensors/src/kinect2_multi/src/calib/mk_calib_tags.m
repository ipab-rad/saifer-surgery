%% extrinsic calibration for multiple kinects
addpath('yaml');
addpath('aprilgrid');
addpath('utils');
%
datapath = [pwd '/data/tags'];
load_saved = false;
export_world = false;
%
tree = rostf;
pause(1);
%tree.AvailableFrames   

%% check ROS license
if ~load_saved
    %rosinit
%     %% depth2cam calib_pose translation from /iai_kinect2/kinect2_bridge/data/<serial> 
%     Text = [ -5.2389868607350183e-02, 1.2874182280404737e-04, -6.0200591093190247e-04; % K1
%            -5.2594798617069928e-02, 1.2749745060958498e-04, 4.4620094844679582e-03;  % K2
%            -4.9942248521624273e-02, 5.3111138034430663e-04, -1.2961172534671949e-02; % K3 
%            -5.1987556513656191e-02, 4.5380178800415600e-04, 1.3901192146455754e-03 ;
%            0                      , 0                     , 0 ];  %RS1
%     % manual adj
%     Text(:,1) = Text(:,1) - 0.04;
%     Text(:,2) = Text(:,2) + 0.04;
%     Text(:,3) = Text(:,3) - 0.02;
    %% capture images and tags
    tags = cell(1,4);
    pcl = cell(1,4);
    figure(112); clf;
    for kid = 1:4
      %% tags
      tx = getTransform(tree, sprintf('kinect_%d_rgb_optical_frame',kid),sprintf('kinect_%d_ir_optical_frame',kid),'Timeout',2); 
      A = eye(4);
      A(1:3,1:3) = quat2rotm([tx.Transform.Rotation.W tx.Transform.Rotation.X tx.Transform.Rotation.Y tx.Transform.Rotation.Z]); 
      A(4,1:3) = [tx.Transform.Translation.X tx.Transform.Translation.Y tx.Transform.Translation.Z];  
     
      [tags{kid}, ID, T, R, V, img] = mk_capture_tags(kid, datapath, affine3d(A));
      
      %% get pcl from camera
      rosTopic = sprintf('/kinect_%d/sd/points',kid);
      sub = rossubscriber(rosTopic);
      fprintf('Waiting for %s msg... ',rosTopic); tic;
      pause(5);
      msg = receive(sub,10);
      pcl{kid} = pointCloud(msg.readXYZ,'Color',msg.readRGB);
      sub.delete();
      toc;
      pcwrite(pcl{kid},sprintf('%s/K%d_points_sd.ply',datapath,kid));

      % transform tags from optical to depth

      
      %% plot
      figure(111); subplot(2,3,kid); imshow(img);
      figure(112); subplot(2,3,kid); cla;
      Rk = eul2rotm(pi*[0 -0.5 0.5]);
      Tk = T*Rk';
      %Tk = (T-repmat(Te(kid,:),size(T,1),1))*Rk';
      plot3(Tk(:,1),Tk(:,2),Tk(:,3),'r*'); grid on; hold on; axis equal; 
      pcshow(pcl{kid}); zlim([0 3]);
      plot3(0,0,0,'go');
      plotCamera('Location',[0 0 0],'Orientation',eul2rotm([0 0 0]),'Size',0.2);
      %
      D = Tk + 0.1*V*Rk';
      line([Tk(:,1) D(:,1)]', [Tk(:,2) D(:,2)]', [Tk(:,3) D(:,3)]'); grid on; hold on; axis equal;
      %
      [N,B,P] = affine_fit(Tk(ID<=36,:));
      plot3(P(1),P(2),P(3),'ro');
      PN = P + 0.5*N';
      line([P(1) PN(1)],[P(2) PN(2)],[P(3) PN(3)],'LineWidth',3);
      zlim([0 3]); xlim([-2 2]); ylim([-2 2]); view(0,-45); drawnow;
    end
    
    %% Realsense camera
    nRS = 1; % Set number of Realsense cameras
    for RSid = kid+1:kid+nRS
      %% tags
      tx = getTransform(tree, sprintf('realsense_%d_color_optical_frame',RSid),sprintf('realsense_%d_depth_optical_frame',RSid),'Timeout',2); 
      A = eye(4);
      A(1:3,1:3) = quat2rotm([tx.Transform.Rotation.W tx.Transform.Rotation.X tx.Transform.Rotation.Y tx.Transform.Rotation.Z]); 
      A(4,1:3) = [tx.Transform.Translation.X tx.Transform.Translation.Y tx.Transform.Translation.Z];  
      
      [tags{RSid}, ID, T, R, V, img] = mk_capture_tags(RSid, datapath, affine3d(A));
      %% get pcl from camera
      rosTopic = sprintf('/realsense_%d/depth/color/points',RSid); % Keep the structure for multiple realsense
      sub = rossubscriber(rosTopic);
      fprintf('Waiting for %s msg... ',rosTopic); tic;
      pause(5);
      msg = receive(sub,10);
      pcl{RSid} = pointCloud(msg.readXYZ,'Color',msg.readRGB);
      sub.delete();
      toc;
      plyPath = sprintf('%s/K%d_points_sd.ply',datapath,RSid);
      pcwrite(pcl{RSid},plyPath); 
      
      %% plot
      figure(111); subplot(2,3,RSid); imshow(img);
      figure(112); subplot(2,3,RSid); cla;
      Rk = eul2rotm(pi*[0 -0.5 0.5]);
      Tk = T*Rk';
      %Tk = (T-repmat(Te(kid,:),size(T,1),1))*Rk';
      plot3(Tk(:,1),Tk(:,2),Tk(:,3),'r*'); grid on; hold on; axis equal; 
      pcshow(pcl{RSid}); zlim([0 3]);
      plot3(0,0,0,'go');
      plotCamera('Location',[0 0 0],'Orientation',eul2rotm([0 0 0]),'Size',0.2);
      %
      D = Tk + 0.1*V;
      line([Tk(:,1) D(:,1)]', [Tk(:,2) D(:,2)]', [Tk(:,3) D(:,3)]'); grid on; hold on; axis equal;
      %
      [N,B,P] = affine_fit(Tk(ID<=36,:));
      plot3(P(1),P(2),P(3),'ro');
      PN = P + 0.5*N';
      line([P(1) PN(1)],[P(2) PN(2)],[P(3) PN(3)],'LineWidth',3);
      zlim([0 3]); xlim([-2 2]); ylim([-2 2]); view(0,-45); drawnow;      
    end
    
else
    %% load saved
    tags{1} = YAML.read(sprintf('%s%sK1_image_color_rect_hd_tags.yml',datapath,slashCharacter));
    tags{2} = YAML.read(sprintf('%s%sK2_image_color_rect_hd_tags.yml',datapath,slashCharacter));
    tags{3} = YAML.read(sprintf('%s%sK3_image_color_rect_hd_tags.yml',datapath,slashCharacter));
    tags{4} = YAML.read(sprintf('%s%sK4_image_color_rect_hd_tags.yml',datapath,slashCharacter));
    tags{5} = YAML.read(sprintf('%s%sK4_image_color_rect_hd_tags.yml',datapath,slashCharacter));
end
   
%% register tags - procrustes
TR = mk_transform_tags(tags);

% refine ICP
%% refine ICP
% if rosLicense
%     TR2 = mk_run_icp(TR, pcl, rosLicense);
% else
%     TR2 = mk_run_icp(TR, [datapath slashCharacter], rosLicense);
% end
%% manual 
% TR{1}.T(1,4) = TR{1}.T(1,4) - 0.03;
% 
% TR{2}.T(1,4) = TR{2}.T(1,4) + 0.05;
% TR{2}.T(2,4) = TR{2}.T(2,4) + 0.15;
% TR{2}.T(3,4) = TR{2}.T(3,4) + 0.02;
% 
% TR{3}.T(1,4) = TR{3}.T(1,4) + 0.13;
% TR{3}.T(2,4) = TR{3}.T(2,4) + 0.10;
% 
% TR{4}.T(1,4) = TR{4}.T(1,4) + 0.02;
% TR{4}.T(2,4) = TR{4}.T(2,4) - 0.06;
% TR{4}.T(3,4) = TR{4}.T(3,4) + 0.02;


%% get robot base to world tf

if export_world
    %% RS5
    blueLink = 'blue_base_link';
    tfBlue = getTransform(tree, blueLink,'gripper_base_link','Timeout',2);  
    qb = tfBlue.Transform.Rotation;
    tb = tfBlue.Transform.Translation;

    %%
    TB = eye(4);
    TB(1:3,1:3) = quat2rotm([qb.W qb.X qb.Y qb.Z]);
    TB(1:3,4) = [tb.X tb.Y tb.Z];

    % If we want to move the robot base to K1, first apply the transform from
    % base to gripper "inv(TB)" and then from gripper to K1 "TR{5}.T". There is
    % a missing transform between RealSense and gripper, which should be in
    % between the tranforms
    T_RS2Grip = eye(4,4);
    TW =(TR{5}.T) * T_RS2Grip * inv(TB); 

    % TW = eye(4);
    % TW(1:3,1:3) = TB(1:3,1:3) * TR{5}.T(1:3,1:3);
    % TW(1:3,4) = TB(1:3,4) + TR{5}.T(1:3,4);

    % export launch file - static transform publisher
    launchpath = [datapath '/kinect2_multi_reg_tf_world.launch'];
    mk_export_calib_world(TR, TW, blueLink, launchpath);
    fprintf('Extrinsic calibration exported to %s.\n',launchpath);
else
    %% export launch file - static transform publisher
    launchpath = [datapath '/kinect2_multi_reg_tf_link.launch'];
    mk_export_calib_link(TR, launchpath);
    fprintf('Extrinsic calibration exported to %s.\n',launchpath);    
end
