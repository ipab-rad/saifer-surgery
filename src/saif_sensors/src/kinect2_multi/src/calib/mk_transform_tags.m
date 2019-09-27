function TR = mk_transform_tags(tags)
%% get ids

reorder = [2,3,1];
scale = 1;

numTags = zeros(1,4);
numViews = size(tags,2);
%% check all tags
if 1 %length(F1) == length(F2) && length(F1) == length(F3) && length(F1) == length(F4)
    
    for nV = 1:numViews
        F = fieldnames(tags{nV});
        markersTrans{nV} = cell(60,1);
        numTags(nV) = length(F);
        for i = 1:numTags(nV)
            %% parse IDs
            name1 = F{i};
            name1 = name1(3:end);

            markersName{nV}{i} = str2num(name1)+1;
            taux = tags{nV}.(F{i}).translation;
            taux = scale*[-taux(reorder(1)) -taux(reorder(2)) taux(reorder(3))];
            markersTrans{nV}{markersName{nV}{i}} = taux;
            markersRot{nV}{markersName{nV}{i}} = tags{nV}.(F{i}).rotation;   
        end
    end

    
    %% Registration
    Z1a = cat(1,markersTrans{1}{:});
    
    %%%% Align data coordinate system with world coordinate system
    %% Vertical orientation
    
    % Find the normal of the board (first 36 targs) and align vertically
   	Z1 = cat(1,markersTrans{1}{1:end});
    gridnorm = affine_fit(Z1);
    if gridnorm(3) > 0
        gridnorm = -gridnorm;
    end
    %r1 = vrrotvec2mat(vrrotvec(-gridnorm,[ 0 0 1 ]));
    r1 = axang2rotm(vrrotvec(-gridnorm,[ 0 0 1 ]));
    r2 = axang2rotm([1 0 0 pi]); %*axang2rotm2([0 0 1 -0.97]);
    
    %% Horizontal orientation
    raux  = r2*r1;
    t1 = [raux -raux*mean(Z1a,1)'; 0 0 0 1];
    Z1aux = transformPcl(Z1a,t1); % Flatten the marker centers to calculate horizontal rotation
    id0 = Z1aux(1,:);
    ang0fix = pi*3/4;
    if isempty(id0) 
       id0 = Z1aux(8,:);
    end
    ang0 = atan2(id0(1),id0(2));   
    r0 = axang2rotm([0 0 1 ang0-ang0fix]); % Calculate rotation between marker 0 angle and 135 degrees
    
    %% World global orientation
    r = r0*r2*r1;
    
    viewFix = 1;
    for viewMove = 2:numViews
        % Find individual transforms between view 1 and each of other views
        
       % [nTag, viewMin] = min([numTags(viewFix),numTags(viewMove)]);
        cont = 1;
        [nameAux] = intersect(cat(1,markersName{viewMove}{:}),cat(1,markersName{viewFix}{:})); % Find common tags
        numMarkers = length(nameAux);
        for i=1:numMarkers
            if ~isempty(markersTrans{viewFix}{nameAux(i)}) & ~isempty(markersTrans{viewMove}{nameAux(i)})
                tagsViewFix{viewMove}(cont,:) = markersTrans{viewFix}{nameAux(i)};
                tagsViewMove{viewMove}(cont,:) = markersTrans{viewMove}{nameAux(i)};  
                cont = cont+1;
            end
        end 
        [d(viewMove),~,tr{viewMove}] = procrustes(tagsViewFix{viewMove},tagsViewMove{viewMove},'reflection',false,'scaling',false);
    end
    
    %% create transforms
    TR = cell(numViews,1);
    
    TR{1}.T = [r -r*mean(Z1,1)'; 0 0 0 1];
    TR{1}.s = eye(4,4);
    TR{1}.c = 'cam1';
    for nV = 2:numViews %first view is fix
        TR{nV}.T = TR{1}.T* [inv(tr{nV}.T) mean(tr{nV}.c,1)'; 0 0 0 1];

        TR{nV}.s = eye(4,4);
        TR{nV}.s = TR{nV}.s * tr{nV}.b;
        TR{nV}.s(4,4) = 1;

        TR{nV}.c = sprintf('cam%d->map',nV);
    end
    
    if 1
        %%
       	Z1 = transformPcl(Z1a,TR{1}.T);
        cc1 = transformPcl([0 0 0],TR{1}.T,TR{1}.s);
        figure(1); clf; set(gcf,'Name','Original');
        hold on; axis image; grid on;
        plot3(cc1(1),cc1(2),cc1(3),'ro');
        plot3(Z1(:,1),Z1(:,2),Z1(:,3),'*');
        for nV = 2:numViews
            Z12 = transformPcl(tagsViewFix{nV},TR{1}.T,TR{1}.s);       
            Z2 = transformPcl(tagsViewMove{nV},TR{nV}.T,TR{nV}.s);
            plot3(Z2(:,1),Z2(:,2),Z2(:,3),'ro');
            line([Z12(:,1) Z2(:,1)]',[Z12(:,2) Z2(:,2)]',[Z12(:,3) Z2(:,3)]');
        end 
        
       
        
        for k = 1:numViews
            %%
            Tx = TR{k}.T;
            R = Tx(1:3,1:3);
            T = Tx(1:3,4);
            plot3(T(1),T(2),T(3),'bx');
            plotCamera('Location',T,'Orientation',R','Opacity',0.1,'Size',0.1,'Label',sprintf('K%d',k));
        end
    end
    %    if 0
    %         f1 = figure;set(f1,'Name','Original');  hold on; axis image;
    %             plot3(0,0,0,'go');
    %             plot3(markersTrans1(:,1),markersTrans1(:,2),markersTrans1(:,3),'r*');
    %             plot3(markersTrans2(:,1),markersTrans2(:,2),markersTrans2(:,3),'b*');
    %             plot3(markersTrans3(:,1),markersTrans3(:,2),markersTrans3(:,3),'g*');
    %             plot3(markersTrans4(:,1),markersTrans4(:,2),markersTrans4(:,3),'k*');
    %
    %             quiver3(markersTrans1(:,1),markersTrans1(:,2),markersTrans1(:,3),markersRot1(:,1),markersRot1(:,2),markersRot1(:,3),'r');
    %             quiver3(markersTrans2(:,1),markersTrans2(:,2),markersTrans2(:,3),markersRot2(:,1),markersRot2(:,2),markersRot2(:,3),'b');
    %             quiver3(markersTrans3(:,1),markersTrans3(:,2),markersTrans3(:,3),markersRot3(:,1),markersRot3(:,2),markersRot3(:,3),'g');
    %             quiver3(markersTrans4(:,1),markersTrans4(:,2),markersTrans4(:,3),markersRot4(:,1),markersRot4(:,2),markersRot4(:,3),'k');
    %
    %             Rotm1 = eul2rot(markersRot1);
    %              for i=1:size(markersRot1,1)
    %                  D(i,:) = markersTrans1(i,:) + (0.1*diag(Rotm1(:,:,i)))';
    %              end
    %             Rotm2 = eul2rot(markersRot2);
    %              for i=1:size(markersRot2,1)
    %                  D2(i,:) = markersTrans2(i,:) + (0.1*diag(Rotm2(:,:,i)))';
    %              end
    %             Rotm3 = eul2rot(markersRot3);
    %              for i=1:size(markersRot3,1)
    %                  D3(i,:) = markersTrans3(i,:) + (0.1*diag(Rotm3(:,:,i)))';
    %              end
    %             Rotm4 = eul2rot(markersRot4);
    %              for i=1:size(markersRot4,1)
    %                  D4(i,:) = markersTrans4(i,:) + (0.1*diag(Rotm4(:,:,i)))';
    %              end
    %              %line([markersTrans1(:,1) D(:,1)]', [markersTrans1(:,2) D(:,2)]', [markersTrans1(:,3) D(:,3)]');
    %              %line([markersTrans2(:,1) D2(:,1)]', [markersTrans2(:,2) D2(:,2)]', [markersTrans2(:,3) D2(:,3)]');
    %              %line([markersTrans3(:,1) D3(:,1)]', [markersTrans3(:,2) D3(:,2)]', [markersTrans3(:,3) D3(:,3)]');
    %              %line([markersTrans4(:,1) D4(:,1)]', [markersTrans4(:,2) D4(:,2)]', [markersTrans4(:,3) D4(:,3)]');
    %
    %              axis image;
    %
    %         f2 = figure('Name','Aligned'); hold on; axis image
    %                     plot3(0,0,0,'go');
    %             plot3(markersTrans1(:,1),markersTrans1(:,2),markersTrans1(:,3),'r*');
    %             plot3(Z2(:,1),Z2(:,2),Z2(:,3),'b*');
    %             plot3(Z3(:,1),Z3(:,2),Z3(:,3),'g*');
    %             plot3(Z4(:,1),Z4(:,2),Z4(:,3),'k*');
    %
    % %             quiver3(markersTrans1(:,1),markersTrans1(:,2),markersTrans1(:,3),markersRot1(:,1),markersRot1(:,2),markersRot1(:,3),'r');
    % %             quiver3(Z2(:,1),Z2(:,2),Z2(:,3),Z2r(:,1),Z2r(:,2),Z2r(:,3),'b');
    % %             quiver3(Z3(:,1),Z3(:,2),Z3(:,3),Z3r(:,1),Z3r(:,2),Z3r(:,3),'g');
    % %             quiver3(Z4(:,1),Z4(:,2),Z4(:,3),Z4r(:,1),Z4r(:,2),Z4r(:,3),'k');
    %
    %             Rotm1 = eul2rot(markersRot1);
    %             for i=1:size(markersRot1,1)
    %                 D(i,:) = markersTrans1(i,:) + (0.1*diag(Rotm1(:,:,i)))';
    %             end
    %
    %             Rotm2 = eul2rot(Z2r);
    %             for i=1:size(Z2r,1)
    %                 D2(i,:) = Z2(i,:) + (0.1*diag(Rotm2(:,:,i)))';
    %             end
    %
    %             Rotm3 = eul2rot(Z3r);
    %             for i=1:size(Z3r,1)
    %                 D3(i,:) = Z3(i,:) + (0.1*diag(Rotm3(:,:,i)))';
    %             end
    %
    %             Rotm4 = eul2rot(Z4r);
    %             for i=1:size(Z4r,1)
    %                 D4(i,:) = Z4(i,:) + (0.1*diag(Rotm4(:,:,i)))';
    %             end
    %              line([markersTrans1(:,1) D(:,1)]', [markersTrans1(:,2) D(:,2)]', [markersTrans1(:,3) D(:,3)]');
    %              line([Z2(:,1) D2(:,1)]', [Z2(:,2) D2(:,2)]', [Z2(:,3) D2(:,3)]');
    %              line([Z3(:,1) D3(:,1)]', [Z3(:,2) D3(:,2)]', [Z3(:,3) D3(:,3)]');
    %              line([Z4(:,1) D4(:,1)]', [Z4(:,2) D4(:,2)]', [Z4(:,3) D4(:,3)]');
    %             axis image
    %     end
    
else
    error('Some tags are missing!');
end