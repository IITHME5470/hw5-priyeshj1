clear
tid =  3185;
a = dlmread(sprintf('T_x_y_%06d.dat', tid));

n = sqrt(size(a,1));
x = a(1:n:n^2,1);
y = a(1:n, 2);
T = reshape(a(:,3), [n, n]);

figure, clf
contourf(x,y,T','LineColor', 'none')
xlabel('x'), ylabel('y'), title(strcat('t = ',sprintf(' %06d',tid)));
xlim([-0.05 1.05]), ylim([-0.05 1.05]), caxis([-0.05 1.05]), colorbar
colormap('jet')
set(gca, 'FontSize', 14)
screen2jpeg(strcat('cont_T_', sprintf('%04d', tid), '.png'))

figure, clf
Tmid = T(:,n/2);
plot(x, Tmid, '-', 'LineWidth', 2)
xlabel('x'), ylabel('T'), title(strcat('Profile along mid-y at t=',sprintf('%06d',tid)))
xlim([-0.05 1.05])
set(gca, 'FontSize', 14)
screen2jpeg(strcat('line_midy_T_', sprintf('%04d', tid), '.png'))
