%% Detect Faces in an Image Using the Frontal Face Classification Model

% Copyright 2015 The MathWorks, Inc.
% modified by Shih-Ting Huang

%% Create a detector object.
    faceDetector = vision.CascadeObjectDetector; 
    
%% Read input image.
    pic_name = 'class1';
    I = imread(['images/', pic_name,'.jpg']);
    
%% Detect faces. 
    % (x, y ,width, height)
    bboxes = step(faceDetector, I);
    [face_n, ~]= size(bboxes);
    
%% Annotate detected faces.
   IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'Face');   
   figure, imshow(IFaces), title('Detected faces'); 

%% Output the detected faces.
if ~exist(pic_name, 'dir')
  mkdir(pic_name);
end

size_row = 48;
size_col = 48;
J_mat = zeros(face_n, size_row * size_col);

 for i = 1:face_n
    J = imcrop(I, bboxes(i,:));
    % RGB to grayscale
    J = rgb2gray(J);
    % resize to 48 * 48
    J = imresize(J, [size_row size_col]);
    % store the new face
    filename = ([pic_name, '/', pic_name, '_',num2str(i),'.jpg']);
    imwrite(J, filename);
    J_bit = reshape(J, 1, size_row * size_col);
    J_mat(i, :) = J_bit;
 end
 
 % output all image data
csvwrite([pic_name, '.csv'], J_mat)
 