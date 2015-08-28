function c = compute_conf(map)
map = map - min(map(:));
dia = 4;
[h, w] = size(map);
[y, x] = find(map == max(map(:)));

x_start = max(1, x - dia);
x_end = min(w, x + dia);
y_start = max(1, y - dia);
y_end = min(h, y + dia);
c = sum(sum(map(y_start:y_end, x_start: x_end))) / sum(map(:));