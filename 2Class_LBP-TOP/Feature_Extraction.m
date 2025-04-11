clc;
clear;

filePath1='F:\MFED224\Data1';
subdir1 = dir(filePath1);

for i = 3 : 24 %3 : 24
    
    i
    
    folder = ['F:\Codes\LBP-TOP\LBP_TOP_4Frames_Feature\',subdir1(i).name,'\'];

    if exist(folder)==0

        mkdir(folder);

    else
        
        disp('dir is exist');
    
    end
    
    filePath2 = [filePath1, '\',subdir1(i).name];
    
    subdir2 = dir(fullfile(filePath2));
    
    for j = 3 : length(subdir2)
        
        j
            
        filePath3  = [filePath2, '\',subdir2(j).name,'\'];
        
        %subdir3 = dir([fullfile(filePath3),'.bmp']);
        subdir3 = dir(strcat(filePath3,'*.bmp'));
        
        Volume_Data = zeros(256, 256, length(subdir3)); 
        
        for k = 1 : length(subdir3)
                      
            filePath4 = [filePath2, '\',subdir2(j).name,'\',subdir3(k).name];
            
            Volume_Data(:, :, k) = im2double(rgb2gray(imread(filePath4))); 
            
        end
        
        Histogram = zeros((length(subdir3)-4)+1, 8*8*256*3);
        
        FrameNum = (length(subdir3)-4)+1
        
        for k = 1 : (length(subdir3)-4)+1
            
            Num = 0;
            
            for p = 1 : 8
                
                for q = 1 : 8
                    
                    Temp = LBPTOP(Volume_Data(1+(p-1)*32:32+(p-1)*32, 1+(q-1)*32:32+(q-1)*32, k:k+4-1), 1, 1, 1, [8,8,8], 1, 1, 1, 0);
                    
                    Histogram(k, Num*256*3 + 1 : Num*256*3 + 256) = Temp(1,:);
                    Histogram(k, Num*256*3 + 256*1 + 1 : Num*256*3 + 256*1 + 256) = Temp(2,:);
                    Histogram(k, Num*256*3 + 256*2 + 1 : Num*256*3 + 256*2 + 256) = Temp(3,:);
                    
                    Num = Num + 1;
                    
                end
                
            end
            
        end
                
        save([folder,subdir2(j).name,'.mat'], 'Histogram');
        
    end
    
end

