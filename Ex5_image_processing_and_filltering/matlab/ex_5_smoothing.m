%% Autonomous Mobile Robots - Exercise 5 (Image Saliency)
close all;
clear all;
clc;

%% Q1.2.c Smoothing in 2D
% The aim of this exercise is understanding how filtering works in 2
% dimensions. To this end, your task is to read and understand the MATLAB
% code provided for smoothing an image and then fill in the missing
% expressions, which will permit the application of the filter to the
% correct portion of the original image. 

% We will be working with the image of Lena that you know from the lecture.

% read image
lena = imread('lena.jpeg');
% visualize the image 
hFig1 = figure(1);
imshow(uint8(lena));
set(hFig1, 'Name', 'Original Lena');
title('Original Lena');    

% convert it into double type
lena = double(lena);
% get the dimensions
height = size(lena, 1);
width  = size(lena, 2);
% output image
filtered_lena = zeros(size(lena)); 

% The filter used for smoothing is the following kernel. 
F = [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1]/256;

% In the lecture, different ways of handling the boundaries of an image
% when applying the filter were discussed. In this exercise we are ignoring
% the boundaries of the original image, by setting the corresponding
% filtered image values to zero.

% compute the offset for the rows and the columns, where the original image 
% boundaries are to be ignored
row_o = floor(size(F,1)/2); 
col_o = floor(size(F,2)/2);
     
% loop through all the pixels of the filtered image that we need to set their value 
% (i.e. ignoring the pixels at the image boundaries)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  TODO:  DEFINE THE LIMITS FOR THE ROW (i) AND COLUMN (j) INDICES     %%
% these are the maximum indices for row and column where the filter will be
% placed
row_limit = height - row_o;        
col_limit = width -  col_o;      
for i = (1+row_o) : row_limit
   for j = (1+col_o) : col_limit
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%     TODO: FILL IN THE EXPRESSIONS                            %%
        % Extract the relevant original-image values (to be stored in    %
        % tempLena) which will be multiplied by the filter values. In    %
        % order to do this, you need to fill in the expressions for the  % 
        % row and column limits of the values to be extracted from lena  %
        % Given your current filter center (i,j) what are the indices of %
        % the left/right/top/bottom filter edges.
        row_min = i - row_o;
        row_max = i + row_o;
        col_min = j - col_o; 
        col_max = j + col_o;
        %%     END OF YOUR INPUT                                        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempLena = double(lena(row_min : row_max, col_min : col_max));
        filtered_lena(i,j) = sum(sum(F .* tempLena));
   end
end 

plotFilteredLena(lena, filtered_lena);

