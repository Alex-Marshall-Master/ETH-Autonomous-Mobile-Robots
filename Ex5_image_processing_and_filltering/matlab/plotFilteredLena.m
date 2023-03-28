function plotFilteredLena(lena, filtered_lena)
          
% display the difference image
hFig2 = figure(2);
diff_image = uint8(abs(lena - filtered_lena));
imshow(diff_image);
set(hFig2, 'Name', 'IMAGE DIFF');
title(hFig2.CurrentAxes, 'Difference Image: abs(Original - Smoothed)');        
    
% display the filtered image
hFig3 = figure(3);
imshow(uint8(filtered_lena));
set(hFig3, 'Name', 'SMOOTHED IMAGE');        
title(hFig3.CurrentAxes, 'Smoothed Lena');