function O = get_gaussain(I)
% I = zeros(46,46);

[m, n] = size(I);
O = single(zeros(m, n));

output_sigma = 2.5;%1.5;
% I(46,21) = 1;
% I(23,23)=1;
[y, x] = find(I == max(I(:)));
% [y, x] = find(I > 0);

[gx, gy] = meshgrid(1:m, 1:n);
for i = 1:length(x)
     O = O + exp(-0.5 * ((((gy - y(i)).^2 + (gx - x(i)).^2) / output_sigma^2)));
%      figure(22);imagesc(O);
end
    

