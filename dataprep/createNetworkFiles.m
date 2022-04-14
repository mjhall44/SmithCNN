function createNetworkFiles(inputDir,outputDir, numSplits, trainandtest)

%settings
plotcombine =0; % plot the data loaded from each file

%check directories end in a slash
if inputDir(length(inputDir)) ~= '\'
   disp('Directory must end in a slash')
end


totaldats = 0; %total number of waves
%loops through input dir
dirname=inputDir;
a = dir(dirname);
celledWaves = cell(numSplits,1);
splitSect = 0; %not save files, but temp sections to reduce memory usage
    for i=3:length(a) %find all subdirectories
        x=a(i);
        disp(['dir: ' char(x.name)])
        chs = str2num(strrep(x.name,'-', ' ')); %uses folder name to find channels
        if isempty(chs) 
            disp('No channels specified, expecting .mats')
        else
            chan = chs(1):chs(2);
        end
        subnev = dir([inputDir x.name]);
        if plotcombine ==1   % 1 means that each file will be plotted
            numInSub = length(3:length(subnev));
            figure
        end
        for j=3:length(subnev) %read in each nev file alphabetically
            nevName = subnev(j);
            disp(['----' char(nevName.name)])
            fullPath=[inputDir x.name '\' nevName.name];

            [~, ~, fExt] = fileparts(fullPath);
            switch lower(fExt) % chooses how to load in files based on extension
              case '.mat'
                % A MAT file, just loads in waves
                structWave = load(fullPath, 'waveData');
                waveData = structWave.waveData;
              case '.nev'
                % An NEV file
                waveData = cleanReadNev(fullPath, chan);
              otherwise  % Under all circumstances SWITCH gets an OTHERWISE!
                error('Unexpected file extension: %s', fExt);
            end

            if plotcombine == 1
                plotidx = j-2;
                subplot(numInSub,1,plotidx)
                plotSpikeNoise(waveData,char(nevName.name))
            end
            if sum(waveData(:,1)) > 0 %only save if spikes are in set
            fWaveCount = size(waveData,1);
            totaldats = totaldats + fWaveCount;
            extraWaves = rem(fWaveCount,numSplits);
            for k = 1:numSplits    % takes equal int random sections of loaded
                splitC = floor(fWaveCount/numSplits);
                if extraWaves > 0
                    splitC = splitC+1;
                    extraWaves = extraWaves-1;
                end
                splitIdx = randperm(size(waveData,1),splitC);
                celledWaves{k,1} = [celledWaves{k,1}; waveData(splitIdx,:)]; %convert to lowersize could be here
                waveData(splitIdx,:) = NaN;
                waveData = waveData(all(~isnan(waveData),2),:); %nans removed
                
                %reduce memory overflow
                %check available RAM, as to not exceed by loading
                celledWaveInfo = whos('celledWaves');
                [~,systemview] = memory; %systemview.PhysicalMemory.Total /1073741824; for gb
                fileMemUse = celledWaveInfo.bytes/systemview.PhysicalMemory.Total;%bytes
                
                if systemview.PhysicalMemory.Available/systemview.PhysicalMemory.Total < 0.10 || fileMemUse > 0.70
                    disp('Saving and clearing to conserve Memory')
                    splitSect = splitSect+1;
                    unqSect= ['tempCelledWaves_s' num2str(splitSect)];
                    save([outputDir unqSect],'celledWaves','-v7.3')
                    clear celledWaves
                    celledWaves = cell(numSplits,1);
                end
            
                
            end %waveData should be empty
            if isempty(waveData)
                disp(['**Finished Unloading** ::' char(nevName.name)])
            else
                error('Waveforms from %s improperly unloaded!', char(nevName.name));
            end
            else
                disp(['Skipped 0 spike file ::' char(nevName.name)])
            end
        end
    end %each cell contains an equal portion from each file inputted, now save

    
    %make file names for saving
    dots = strfind(fullPath, '.');
    backslash = strfind(fullPath, '\');
    filename = char(extractBetween(fullPath, backslash(length(backslash))+1, dots(length(dots))-1 ));
    
    rootName = filename;
    if trainandtest == 1
        mkdir([outputDir,'training_files'])
        mkdir([outputDir,'testing_files'])
        for k =1:numSplits
            if splitSect >0 % if broken apart to save memory
                for s=1:splitSect
                    unqSect= ['tempCelledWaves_s' num2str(s)];
                    sectObject = matfile([outputDir unqSect]);
                    celledSect = sectObject.celledWaves(k,1);
                    celledWaves{k,1} = [celledWaves{k,1}; celledSect{1,1}];
                    clear celledSect
                end
            end
            unqName= [rootName '_p' num2str(k)];
            waveData = celledWaves{k,1};
            rngVals = randperm(length(waveData),round(length(waveData)*0.2));
            waveDataTest = waveData(rngVals,:);
            waveData(rngVals,:) = NaN;
            waveData = waveData(~isnan(waveData(:,1)),:);
            if k == 0
            save([outputDir '\training_files\train_data'],'waveData','-v7.3')
            waveData = waveDataTest;
            save([outputDir '\testing_files\test_data'],'waveData','-v7.3')
            else
            save([outputDir '\training_files\train_' num2str(k)],'waveData','-v7.3')
            waveData = waveDataTest;
            save([outputDir '\testing_files\test_' num2str(k)],'waveData','-v7.3')
            end
            clear celledWaves{k,1}; waveData=[]; %memory reduction
        end
        clear waveData;
        fprintf("%d training file(s) saved to %s\n",numSplits,[outputDir '\training_files'])
        fprintf("%d testing  file(s) saved to %s\n",numSplits,[outputDir '\testing_files'])
    else
        mkdir([outputDir,'training_files'])
        for k =1:numSplits
            if splitSect >0 % if broken apart to save memory
                for s=1:splitSect
                    unqSect= ['tempCelledWaves_s' num2str(s)];
                    sectObject = matfile([outputDir unqSect]);
                    celledSect = sectObject.celledWaves(k,1);
                    celledWaves{k,1} = [celledWaves{k,1}; celledSect{1,1}];
                    clear celledSect
                end
            end
            unqName= [rootName '_p' num2str(k)];
            waveData = celledWaves{k,1};
            save([outputDir '\training_files\train_' num2str(k)],'waveData','-v7.3')
            clear celledWaves{k,1}; waveData=[]; %memory reduction
        end
        clear waveData;
        fprintf("%d training file(s) saved to %s\n",numSplits,[outputDir '\training_files'])
    end
    fprintf("%d total number of waves used\n",totaldats)
end


function plotSpikeNoise(waveData, nevChar, waveidxs) %plots spike and noise colorcoded on one plot
 samp = length(waveData(1,:))-1;
        labels = waveData(:,1); %assumed 0 and 1
        if ~exist('nevChar', 'var')
            nevChar = 'Sample Waveforms';
        end
        if ~exist('fixed', 'var') % choose the same labels to plot or a random subset
            rngVals = randperm(length(labels),500);
        else
            rngVals = waveidxs;
        end
        logicalrngSpike = rngVals(labels(rngVals,:) == 1);
        logicalrngNoise = rngVals(labels(rngVals,:) == 0);
        
        
        perSpike = round(sum(labels ==1)/length(labels),3)*100;
        perStr = strcat(num2str(perSpike),'%');

        if size(logicalrngNoise) > 0
            bp= plot(1:samp,waveData(logicalrngNoise,2:(samp+1)),'Color', [0.5 0.5 0.5]);
        end
        if size(logicalrngSpike) >0
            hold on
            ap = plot(1:samp, waveData(logicalrngSpike,2:(samp+1)),'Color','g');
            sleg= {'Spike' ,'Noise'};
            if logicalrngNoise > 0
                y =[ap(1), bp(1)]; %this actually works for legend
                legend(y,sleg)
            end
            title(strrep(nevChar,'_', ' '))
            %title(strcat(strrep(nevChar,'_', ' '),[' - ' perStr ' spikes']))%, SNR: ' num2str(round(z,2))]))
        else
           %title(strcat(strrep(nevChar,'_', ' '),' has NO Spikes!'))
           title(strcat(strrep(nevChar,'_', ' '),''))
        end
        xlim([1 52])
        %set(gca,'XTick',[], 'YTick', [])
        ylabel("\muV")
        xlim([1 samp])
        ylim([-150 150])
        hold off
        axis on
end