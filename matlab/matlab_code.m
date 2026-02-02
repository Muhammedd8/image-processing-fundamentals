%% Goruntu Yukleme
I = imread("images/lena.jpg");

%% Griye Cevirme
gri = rgb2gray(I);
figure, imshow(gri), title("Gri Goruntu");

%% Esikleme (Otsu)
level = graythresh(gri);
bw = imbinarize(gri, level);
figure, imshow(bw), title("Esikleme");

%% Negatif Goruntu
negatif = imcomplement(gri);
figure, imshow(negatif), title("Negatif Goruntu");

%% Histogram
figure, imhist(gri), title("Histogram");

%% Histogram Esitleme
hist_eq = histeq(gri);
figure, imshow(hist_eq), title("Histogram Esitleme");

%% Mean Filtresi
w_mean = ones(3) / 9;
mean_img = imfilter(gri, w_mean, 'replicate');
figure, imshow(mean_img), title("Mean Filter");

%% Gaussian Filtresi
w_gauss = fspecial('gaussian', [3 3], 1);
gauss_img = imfilter(gri, w_gauss, 'replicate');
figure, imshow(gauss_img), title("Gaussian Filter");

%% Median Filtresi (Salt & Pepper Gurultu ile)
noise_img = imnoise(gri, 'salt & pepper', 0.1);
median_img = medfilt2(noise_img, [3 3]);

figure;
subplot(1,3,1), imshow(gri), title("Orijinal");
subplot(1,3,2), imshow(noise_img), title("Gurultulu");
subplot(1,3,3), imshow(median_img), title("Median Filtre");

%% Laplacian Filtresi
w_lap = fspecial('laplacian', 0);
lap_img = imfilter(gri, w_lap, 'replicate');
figure, imshow(lap_img), title("Laplacian Filter");

%% Sobel Filtresi
Gx = fspecial('sobel');
Gy = Gx';

sobel_x = imfilter(gri, Gx, 'replicate');
sobel_y = imfilter(gri, Gy, 'replicate');
sobel_mag = uint8(sqrt(double(sobel_x).^2 + double(sobel_y).^2));

figure, imshow(sobel_mag), title("Sobel Kenar Algilama");

%% Goruntu Dondurme ve Oteleme
rot = imrotate(gri, -30);
shift_right = imtranslate(gri, [90 0]);
shift_up = imtranslate(gri, [0 -50]);

figure;
subplot(1,3,1), imshow(rot), title("30° Dondurme");
subplot(1,3,2), imshow(shift_right), title("Saga Oteleme");
subplot(1,3,3), imshow(shift_up), title("Yukari Oteleme");

%% Yeniden Boyutlandirma
img_small = imresize(gri, [100 150]);
img_zoom_out = imresize(gri, 0.7, 'bicubic');
img_zoom_in = imresize(gri, 1.5, 'bicubic');

figure;
subplot(1,3,1), imshow(img_small), title("100x150");
subplot(1,3,2), imshow(img_zoom_out), title("0.7 Oran");
subplot(1,3,3), imshow(img_zoom_in), title("1.5 Oran");

%% Morfolojik Islemler
se = strel('disk', 3);

dilate_img = imdilate(bw, se);
erode_img = imerode(bw, se);
open_img = imopen(bw, se);
close_img = imclose(bw, se);

figure;
subplot(2,2,1), imshow(dilate_img), title("Yayma");
subplot(2,2,2), imshow(erode_img), title("Asindirma");
subplot(2,2,3), imshow(open_img), title("Acma");
subplot(2,2,4), imshow(close_img), title("Kapama");
