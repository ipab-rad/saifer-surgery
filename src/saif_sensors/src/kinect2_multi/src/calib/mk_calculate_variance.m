function [CentreMatrix, CentreMean, DKinect] = mk_calculate_variance(CentreSphereCorrected)

    CentreMatrix(:,:,1) = CentreSphereCorrected{1};
    CentreMatrix(:,:,2) = CentreSphereCorrected{2};
    CentreMatrix(:,:,3) = CentreSphereCorrected{3};
    CentreMatrix(:,:,4) = CentreSphereCorrected{4};

    CentreMean = mean(CentreMatrix,3);

    for i = 1:size(CentreMatrix,2)
        DKinect(1,i) = sqrt((CentreMatrix(1,i,1)-CentreMean(1,i)).^2 + (CentreMatrix(2,i,1)-CentreMean(2,i)).^2 + (CentreMatrix(3,i,1)-CentreMean(3,i)).^2);
        DKinect(2,i) = sqrt((CentreMatrix(1,i,2)-CentreMean(1,i)).^2 + (CentreMatrix(2,i,2)-CentreMean(2,i)).^2 + (CentreMatrix(3,i,2)-CentreMean(3,i)).^2);
        DKinect(3,i) = sqrt((CentreMatrix(1,i,3)-CentreMean(1,i)).^2 + (CentreMatrix(2,i,3)-CentreMean(2,i)).^2 + (CentreMatrix(3,i,3)-CentreMean(3,i)).^2);
        DKinect(4,i) = sqrt((CentreMatrix(1,i,4)-CentreMean(1,i)).^2 + (CentreMatrix(2,i,4)-CentreMean(2,i)).^2 + (CentreMatrix(3,i,4)-CentreMean(3,i)).^2);
    end

end