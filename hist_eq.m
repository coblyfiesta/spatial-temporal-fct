%% This function performs the histogram equalization of a colour image
%%% input image i can be any RGB color image
%% output image k1 will be histogram equalised image
function [k1]=hist_eq(i)
 for j=1:3
k=i(:,:,j);
k1(:,:,j)= histeq(k);
 end
%figure, imshow(k1)