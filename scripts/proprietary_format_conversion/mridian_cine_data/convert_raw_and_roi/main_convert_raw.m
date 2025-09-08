% Code to convert a single .raw MRIdian file containing a series of cine MRI frames to separate .mha files
% Tested with MATLAB 2019b

% Input and output paths
path_to_input_raw = '/your_path/data_test/image_data.raw';
path_to_output_folder = '/your_path/converted_frames/';

% Create output dir
mkdir(path_to_output_folder)

% Open the raw file in read mode
fid = fopen(path_to_input_raw, 'r');  

% Read the header information
nx = fread(fid, 1, 'int');  % size [pixels]
ox = fread(fid, 1, 'double'); % location
dx = fread(fid, 1, 'double'); % pixel spacing [cm]
ny = fread(fid, 1, 'int');
oy = fread(fid, 1, 'double');
dy = fread(fid, 1, 'double');
nz = fread(fid, 1, 'int');
oz = fread(fid, 1, 'double');
dz = fread(fid, 1, 'double');

nr_of_frames = fread(fid, 1, 'int16');

current_frame = 1;  % assumes cine MRI starts with frame=1
while ~feof(fid)    
    % Read the image data
    image = fread(fid, nx*ny, 'int16');
    image_reshaped = reshape(image, nx, ny);
    % Switch rows and columns, effectively rotates by 90
    image_reshaped_permuted = permute(image_reshaped, [ 2 1 ]);
    % Flip from left to right
    % image_reshaped_permuted_flipped = flip(image_reshaped_permuted, 2);

    % Plot current frame
%     imagesc(image_reshaped_permuted, [0 1000]);
%     colormap gray;
%     colorbar;
%     axis image;
%     pause(0.1);
    
    % Create file name for current frame 
    file_name = sprintf('%s_%s%s','frame',num2str(current_frame,'%05.f'), '.mha')
    path_to_output = sprintf( '%s%s%s', path_to_output_folder, '/', file_name );
    
    % Save as mha with dimensions (h,w,1)
    WriteMha(path_to_output, image_reshaped_permuted, [ox oy oz], [dx*10 dy*10 dz*10], 'float');
    
    current_frame = current_frame + 1;
    if current_frame == nr_of_frames + 1
        break
    end
end


