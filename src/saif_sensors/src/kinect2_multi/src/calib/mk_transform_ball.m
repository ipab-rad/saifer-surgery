function TR = mk_transform_ball(Points,T1)
    %T1 = [eye(3) [0 0 0]'; 0 0 0 1];
    %nV1 & nV2
    viewFix = 1;
    viewMove = 2;
    [~,~,tr2] = procrustes(Points{viewFix}',Points{viewMove}','reflection',false);    

    %nV1 & nV3
    viewFix = 1;
    viewMove = 3;
	[~,~,tr3] = procrustes(Points{viewFix}',Points{viewMove}','reflection',false);
        
     %nV1 & nV4
    viewFix = 1;
    viewMove = 4;
    [~,~,tr4] = procrustes(Points{viewFix}',Points{viewMove}','reflection',false);
    
    %% create transforms
    TR = cell(4,1);
    
    TR{1}.T = T1;
    TR{2}.T = TR{1}.T* [inv(tr2.T) mean(tr2.c,1)'; 0 0 0 1];
    TR{3}.T = TR{1}.T* [inv(tr3.T) mean(tr3.c,1)'; 0 0 0 1];
    TR{4}.T = TR{1}.T* [inv(tr4.T) mean(tr4.c,1)'; 0 0 0 1];

%     TR{2}.T = TR{1}.T* [(tr2.T) [0 0 0]' ; mean(tr2.c,1) 1];
% 	TR{3}.T = TR{1}.T* [(tr3.T) [0 0 0]' ; mean(tr3.c,1) 1];
% 	TR{4}.T = TR{1}.T* [(tr4.T) [0 0 0]' ; mean(tr4.c,1) 1];
    
    TR{1}.s = eye(4,4);
    
    TR{2}.s = eye(4,4);
    TR{2}.s = TR{2}.s * tr2.b;
    TR{2}.s(4,4) = 1;
    
    TR{3}.s = eye(4,4);
    TR{3}.s = TR{3}.s * tr3.b;
    TR{3}.s(4,4) = 1;
    
    TR{4}.s = eye(4,4);
    TR{4}.s = TR{4}.s * tr4.b;
    TR{4}.s(4,4) = 1;
    
    TR{1}.c = 'cam1';
    TR{2}.c = 'cam1<-cam2';
    TR{3}.c = 'cam1<-cam3';
    TR{4}.c = 'cam1<-cam4';    
    
end