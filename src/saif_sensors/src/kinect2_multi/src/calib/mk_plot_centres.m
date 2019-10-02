function mk_plot_centres(CentreMatrix, CentreMean, PointsCorrected)
    figure; hold on;
        plot3(CentreMatrix(1,:,1)',CentreMatrix(2,:,1)',CentreMatrix(3,:,1)','.b');
        plot3(CentreMatrix(1,:,2)',CentreMatrix(2,:,2)',CentreMatrix(3,:,2)','.r');
        plot3(CentreMatrix(1,:,3)',CentreMatrix(2,:,3)',CentreMatrix(3,:,3)','.g');
        plot3(CentreMatrix(1,:,4)',CentreMatrix(2,:,4)',CentreMatrix(3,:,4)','.c');

        quiver3(CentreMatrix(1,:,1)',CentreMatrix(2,:,1)',CentreMatrix(3,:,1)', (CentreMatrix(1,:,1)-CentreMean(1,:))' , (CentreMatrix(2,:,1)-CentreMean(2,:))' , (CentreMatrix(3,:,1)-CentreMean(3,:))',0,'b');
        quiver3(CentreMatrix(1,:,2)',CentreMatrix(2,:,2)',CentreMatrix(3,:,2)', (CentreMatrix(1,:,2)-CentreMean(1,:))' , (CentreMatrix(2,:,2)-CentreMean(2,:))' , (CentreMatrix(3,:,2)-CentreMean(3,:))',0,'r');
        quiver3(CentreMatrix(1,:,3)',CentreMatrix(2,:,3)',CentreMatrix(3,:,3)', (CentreMatrix(1,:,3)-CentreMean(1,:))' , (CentreMatrix(2,:,3)-CentreMean(2,:))' , (CentreMatrix(3,:,3)-CentreMean(3,:))',0,'g');
        quiver3(CentreMatrix(1,:,4)',CentreMatrix(2,:,4)',CentreMatrix(3,:,4)', (CentreMatrix(1,:,4)-CentreMean(1,:))' , (CentreMatrix(2,:,4)-CentreMean(2,:))' , (CentreMatrix(3,:,4)-CentreMean(3,:))',0,'c');

        %Plot average
        plot3(CentreMean(1,:)',CentreMean(2,:)',CentreMean(3,:)','*y');
        plot3(CentreMean(1,:)',CentreMean(2,:)',CentreMean(3,:)','ok');
        axis image

    %Plot Point cloud
    
    figure; hold on
        pcshow(PointsCorrected{1});
        pcshow(PointsCorrected{2});
        pcshow(PointsCorrected{3});
        pcshow(PointsCorrected{4});
    
    
end