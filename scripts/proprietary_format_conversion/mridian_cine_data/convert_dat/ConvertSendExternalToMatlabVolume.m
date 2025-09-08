function [volume] = ConvertSendExternalToMatlabVolume(PathInput, PathOutput, displayModulo, writeMHAorRaw, OS)

volume = [];

% set slash according to operating system
if strcmp(OS, 'Windows')
    slash='\';
else
    slash='/';
end

try
    fid = fopen( PathInput, 'rb' );
    
    numberOfImages = 0;
    
    %     filesize=dir(filename).bytes;
    %     [xSize,ySize,HeaderSize] = getsizes( fid )
    %     sizeimage=xSize*ySize*2+HeaderSize*1+8
    %     maxSlice=filesize/sizeimage    
    
    
    itime = 0;
    while ~feof(fid)
        %tic
        itime = itime+1;
        
        
        % How big is the header.
        
        % This is many ascii strings of dicom data.
        
        % We will parse for rows and columns
        
        headerSize = fread( fid, 1, 'int32' );
        
        
        
        % File ends when they write a zero header size
        
        if headerSize == 0, break, end;
        
        % Or the header comes back with nothing in it.
        
        if isempty(headerSize)
            beep 
            break 
        end
        
        
        
        % How big is the data.
        
        dataSize = fread( fid, 1, 'int32' );
        
        
        
        % Read the ascii header data
        
        header = fread( fid, headerSize, 'int8' );
        
        
        [asciiDicomTags, count] = sscanf( char(header), '%s' );
        
        
        
        % Find number of rows in the header
        
        rowStringLocation = strfind( asciiDicomTags, 'DICOM.NoOfRows' );
        
        rowStringLength = 15; % DICOM.NoOfRols=
        
        firstChar = rowStringLocation + rowStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        ysizeChar = asciiDicomTags( firstChar: lastChar );
        
        ysize = sscanf( ysizeChar, '%d' );
        
        
        
        % Find number of columns in the header
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.NoOfCols' );
        
        colStringLength = 15; % DICOM.NoOfCols=
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        xsize = sscanf( xsizeChar, '%d' );
        
        
        
        % DICOM.PosVec.0
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.PosVec.0' );
        
        colStringLength = 15; % length(DICOM.PosVec.0=)
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        xloc(itime) = sscanf( xsizeChar, '%f' );
        
        
        
        % DICOM.PosVec.1
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.PosVec.1' );
        
        colStringLength = 15; % length(DICOM.PosVec.1=)
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        yloc(itime) = sscanf( xsizeChar, '%f' );
        
        
        
        % DICOM.PosVec.2
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.PosVec.2' );
        
        colStringLength = 15; % length(DICOM.PosVec.2=)
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        zloc(itime) = sscanf( xsizeChar, '%f' );
        
        
        
        % DICOM.SliceThickness = 5.000000
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.SliceThickness' );
        
        colStringLength = 21;
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        sliceThick(itime) = sscanf( xsizeChar, '%f' );
        
        
        
        % DICOM.SliceLocation = 0.000000
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.SliceLocation' );
        
        colStringLength = 20;
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        sliceLocation(itime) = sscanf( xsizeChar, '%f' );
        
        
        
        % DICOM.PixelSpacing.0 = 3.515625
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.PixelSpacing.0' );
        
        colStringLength = 21;
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        pixelSpacing0(itime) = sscanf( xsizeChar, '%f' );
        
        
        
        % DICOM.PixelSpacing.1 = 3.515625
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.PixelSpacing.1' );
        
        colStringLength = 21;
        
        firstChar = colStringLocation + colStringLength;
        
        lastChar = firstChar + 5;
        
        if lastChar > length( asciiDicomTags )
            
            lastChar = length( asciiDicomTags );
            
        end
        
        xsizeChar = asciiDicomTags( firstChar: lastChar );
        
        pixelSpacing1(itime) = sscanf( xsizeChar, '%f' );
        
        
        
        colStringLocation = strfind( asciiDicomTags, 'DICOM.SequenceDescription' );
        
        colStringLength = 26;
        
        firstChar = colStringLocation + colStringLength;
        
        if firstChar ~= []
            
            nchars = strfind( asciiDicomTags( firstChar: end ), 'DICOM.' );
            
            lastChar = firstChar + nchars(1) - 2;
            
            protocolName = asciiDicomTags( firstChar : lastChar );
            
        end
        
        
        
%         if mod( itime, 100 ) == 0
            % print some info
%             fprintf( 'time=%05d dimensions = %d %d position = %e %e %e %e %e %e\n',itime, ysize, xsize,xloc(itime), yloc(itime), zloc(itime), sliceLocation(itime), pixelSpacing0(itime), pixelSpacing1(itime) );  
%         end
        
        % record elapsed time (uncomment tic @ line 26 and toc @ next line 
        % to print elapses times)
        % toc
        
        
        % If this is the first image, we must allocate the volume
        
        if numberOfImages == 0
            
            volume(1, 1:ysize, 1:xsize) = 0;
            
            numberOfImages = numberOfImages + 1;
            
        else
            
            numberOfImages = numberOfImages + 1;
            
        end
        
        
        
        image = fread( fid, xsize*ysize, 'int16' );
        
        image = reshape( image, ysize, xsize );
        
        % flip from left to right
        image = flip(image, 2);
        
        
        volume(numberOfImages, :,:) = image;
        
        
        
        % Display input image every so often.
        
        mod100 = mod( numberOfImages, displayModulo );

        if mod100 == 0

            % imagesc(image);

            imagesc( image, [0 1000] );

            title( ['time=' num2str(itime) ] );

            colormap gray;

            colorbar;

            axis image;

            pause(0.1);



        end


        
    end
    
    
    
    fclose(fid);
    
    
    % write to raw3 format
    if writeMHAorRaw == 1
        
        fid = fopen ( [PathInput '.raw3' ], 'wb' );
        
        nsize = size( volume );
        
        
        
        nx = nsize(1);
        
        ny = nsize(2);
        
        nz = nsize(3);
        
        
        
        dx = pixelSpacing0(1);
        
        dy = pixelSpacing1(1);
        
        dz = sliceThick(1);
        
        
        
        ox = xloc(1);
        
        oy = yloc(1);
        
        oz = zloc(1);
        
        
        
        fwrite( fid, nx, 'int' );
        
        fwrite( fid, dx, 'double' );
        
        fwrite( fid, ox, 'double' );
        
        
        
        fwrite( fid, ny, 'int' );
        
        fwrite( fid, dy, 'double' );
        
        fwrite( fid, oy, 'double' );
        
        
        
        fwrite( fid, nz, 'int' );
        
        fwrite( fid, dz, 'double' );
        
        fwrite( fid, oz, 'double' );
        
        
        disp('Writing to raw3... \n')
        fwrite( fid, int16(volume), 'int16' );
  
        fclose(fid);
        
    % save volume in .mha format
    else
        starting_frame = 6;  % start from 6th frame as the 5 previous ones are not present in corresponding OGV cine MRI
        for frame_number=starting_frame:1:numberOfImages
            % concatenate string with number frame_frame_number
            filename3D = sprintf('%s_%s%s','frame',num2str(frame_number,'%05.f'),'.mha');
            file_frame = sprintf( '%s%s%s', PathOutput, slash, filename3D );
            
            fprintf('Writing to %s \n', file_frame)
            WriteMha(file_frame, volume(frame_number,:,:),[zloc(frame_number) yloc(frame_number) xloc(frame_number)],[sliceThick(frame_number) pixelSpacing1(frame_number) pixelSpacing0(frame_number) ], 'float');
        end
    end
    
    
catch ME
    
    
    
    fprintf( '%s\n', ME.message );
    
    
    
end



end