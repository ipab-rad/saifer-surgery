function mk_export_calib_link(TR, launchPath )
tree = rostf;
%% export ros launch file for TF
fid = fopen(launchPath,'w');
%% global tx
GTx = eye(4);
%GTx(1:3,1:3) = eul2rotm([0 0 -pi/2]);
%GTx(1:3,4) = [0 0 0.7];
%GTx(1:3,1:3) = eul2rotm([0 0 -2.1]);
%GTx(1:3,4) = [-1 -1 1];
%%
fprintf(fid,'<launch>\n');
%figure; hold on; grid on; axis equal; plot3(0,0,0,'go');
for kid = 1:4
    %% get back to link
    tx = getTransform(tree, sprintf('kinect_%d_ir_optical_frame',kid),sprintf('kinect_%d_link',kid),'Timeout',2);
    A = eye(4);
    A(1:3,1:3) = quat2rotm([tx.Transform.Rotation.W tx.Transform.Rotation.X tx.Transform.Rotation.Y tx.Transform.Rotation.Z]);
    A(1:3,4) = [tx.Transform.Translation.X tx.Transform.Translation.Y tx.Transform.Translation.Z];
    %%
    Tx = GTx * A * (TR{kid}.T);
    R = Tx(1:3,1:3);
    T = Tx(1:3,4);
    %plot3(T(1),T(2),T(3),'bx');
    %plotCamera('Location',T,'Orientation',R','Opacity',0.1,'Size',0.1,'Label',sprintf('K%d',k));
    Q = rotm2quat(R); % form q = [w x y z], with w as the scalar number.
    spose = sprintf('%f %f %f %f %f %f %f',T(1),T(2),T(3),Q(2),Q(3),Q(4),Q(1)); % x y z qx qy qz qw
    fprintf(fid,'  <node pkg="tf" type="static_transform_publisher" name="kinect_%d_tfs" args="%s map kinect_%d_link 50"/>\n',kid,spose,kid);
    % static_transform_publisher x y z qx qy qz qw frame_id child_frame_id  period_in_ms
end

%% realsense
for kid = 5:length(TR)
    %% get back to link
    tx = getTransform(tree, sprintf('realsense_%d_depth_optical_frame',kid),sprintf('realsense_%d_link',kid),'Timeout',2);
    A = eye(4);
    A(1:3,1:3) = quat2rotm([tx.Transform.Rotation.W tx.Transform.Rotation.X tx.Transform.Rotation.Y tx.Transform.Rotation.Z]);
    A(1:3,4) = [tx.Transform.Translation.X tx.Transform.Translation.Y tx.Transform.Translation.Z];
    %%
    Tx = GTx * A * (TR{kid}.T);
    R = Tx(1:3,1:3);
    T = Tx(1:3,4);
    %plot3(T(1),T(2),T(3),'bx');
    %plotCamera('Location',T,'Orientation',R','Opacity',0.1,'Size',0.1,'Label',sprintf('K%d',k));
    Q = rotm2quat(R); % form q = [w x y z], with w as the scalar number.
    spose = sprintf('%f %f %f %f %f %f %f',T(1),T(2),T(3),Q(2),Q(3),Q(4),Q(1)); % x y z qx qy qz qw
    fprintf(fid,'  <node pkg="tf" type="static_transform_publisher" name="realsense_%d_tfs" args="%s map realsense_%d_link 50"/>\n',kid,spose,kid);
    % static_transform_publisher x y z qx qy qz qw frame_id child_frame_id  period_in_ms
    
end


fprintf(fid,'</launch>\n');
%
fclose(fid);