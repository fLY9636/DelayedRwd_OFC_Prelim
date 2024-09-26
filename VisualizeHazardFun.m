edges = 5:0.1:12;
Lumped_ToneT = [];
for i = 1:100000
    ToneT = exprnd(4.5)+5;
    if ToneT <= 12 && ToneT >= 5
        Lumped_ToneT(end+1, 1) = ToneT;
    end
end
[N, edges1] = histcounts(Lumped_ToneT, edges, 'Normalization', 'probability');
N_auc = N/trapz(edges1(1:end-1), N);
N_fit = fit(edges1(1:end-1)', N_auc', 'exp1');
N_smooth = N_fit(edges1(1:end-1)');

CDF0 = cumtrapz(edges1(1:end-1), N_smooth);

figure;
subplot(4, 1, 1); hold on
plot(edges(1:end-1), N_smooth)
subplot(4, 1, 2); hold on
plot(edges(1:end-1), CDF0)
subplot(4, 1, 3); hold on
plot(edges(1:end-1), N_smooth./(1-CDF0))
ylim([0 1])

%%
% Design your Gaussian kernel
sigma = 5;  % Standard deviation of the Gaussian kernel
kernel_size = 2 * ceil(3 * sigma) + 1;  % Kernel size (3 standard deviations on each side)

% Create the Gaussian kernel
x_kernel = linspace(-kernel_size / 2, kernel_size / 2, kernel_size);
gaussian_kernel = exp(-x_kernel .^ 2 / (2 * sigma ^ 2));
gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);  % Normalize the kernel

figure
subplot(4, 1, 1)
plot(gaussian_kernel)
% Apply convolution
blurred_PDF = conv(N_smooth, gaussian_kernel, 'same');
blurred_CDF = cumtrapz(edges1(1:end-1), blurred_PDF);
subplot(4, 1, 2)
plot(edges1(1:end-1), blurred_PDF)
subplot(4, 1, 3)
plot(edges1(1:end-1), blurred_CDF)
subplot(4, 1, 4)
plot(edges1(1:end-1), blurred_PDF./(1-blurred_CDF))







