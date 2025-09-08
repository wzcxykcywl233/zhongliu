% Code to recursively convert cine MRI from .dat to .mha 
% using ConvertSendExternalToMatlabVolume function provided by Viewray. The
% function was modifed to correctly orient the cine frames and save as from
% frame nr. 6, as the previous ones are not present in the corresponding OGV
% cine MRIs (steady state magnetisation still building up)


% Define the path you want to search for .dat files in
path_data = './data_test';

% Parameters for conversion
% only display every displayModulo frames
% displayModulo = 1;
displayModulo = 10000000000; % basically never display
% save data as MHA (0) or Raw3 (1) format
writeMHAorRaw = 0;
% operating system, used to set slash direction for saving
OS = 'Linux';   % Windows or Linux
% delete dump and dat files after conversion
delete_dump_dat = false;

% Call the recursive function to search for .dat files in all subfolders
findAndConvertDatFiles(path_data, displayModulo, writeMHAorRaw, OS, delete_dump_dat);



% Define a recursive function to search for .dat files in all subfolders
function findAndConvertDatFiles(folder, displayModulo, writeMHAorRaw, OS, delete_dump_dat)
    % Use the built-in MATLAB function dir() to get a list of all files and folders in the folder
    dirList = dir(folder);

    % Loop over each item in the dirList
    for i = 1:length(dirList)
        % Check if the item is a folder (and not '.' or '..')
        if dirList(i).isdir && ~strcmp(dirList(i).name,'.') && ~strcmp(dirList(i).name,'..')
            % Use recursion to call this same function on the subfolder
            % This will continue looping through all subfolders until we've searched every file in the path
            findAndConvertDatFiles(fullfile(folder,dirList(i).name), displayModulo, writeMHAorRaw, OS, delete_dump_dat);
            
        % If the item is a file, check if it ends in 'sag.dat'  -> cine MRI
        elseif ~dirList(i).isdir && endsWith(dirList(i).name,'sag.dat')
            % Use fullfile() to get the full path to the file
            path_file_original = fullfile(folder,dirList(i).name);
            % Print the path to the .dat file
            fprintf('\n Found cine MRI: %s\n', path_file_original);
            % Get path to .dat without .dat ending
            path_file_saving = path_file_original(1:end-4);
            
            % if folder with mha exists delete it and create new one
            if isfolder(path_file_saving)
                rmdir(path_file_saving,'s')
            end
            mkdir(path_file_saving)
            
            % call Viewray provided function to convert raw format and save
            volume = ConvertSendExternalToMatlabVolume(path_file_original,path_file_saving,displayModulo,writeMHAorRaw,OS);
        
            if delete_dump_dat == true
                delete(path_file_original)
            end
        end
            
        % If the item is a file, check if it ends in '.dat'
        if ~dirList(i).isdir && (endsWith(dirList(i).name,'.dat') | endsWith(dirList(i).name,'.dump'))
            % Use fullfile() to get the full path to the file
            path_file_original = fullfile(folder,dirList(i).name);
            
            if delete_dump_dat == true
                delete(path_file_original)
            end
        % If the item is a file, check if it ends in '.dump'
         if ~dirList(i).isdir && endsWith(dirList(i).name,'.dump')
             % Use fullfile() to get the full path to the file
             path_file_original = fullfile(folder,dirList(i).name);
             
             if delete_dump_dat == true
                 delete(path_file_original)
             end    
        end
    end
end





























