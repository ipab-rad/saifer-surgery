function viewOut = transformPcl(views,transformations,transformScale)
% Function to transform the views
scaling = 0;
if nargin > 2
    scaling = 1;
end
viewOut=views;
%%
sizePoints=size(viewOut,1);      
data=[viewOut ones(sizePoints,1)]'; % To process using the affine transformation
%data=data'*transformations;
data = transformations*data;
if scaling 
    data = data'*transformScale;
    data = data';
end

viewOut = data(1:3,:)';

end