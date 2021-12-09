function createFilesforClassification(inputDir,outputDir)

%settings
plotcombine =0; % plot the data loaded from each file

%check directories end in a slash
totaldats = 0; %total number of waves
%loops through input dir
dirname=inputDir;
a = dir(dirname);
fileCount = 0; %not save files, but temp sections to reduce memory usage
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
            
        %save file
        backs = strfind(nevName.name, '.');
        filename = char(extractBetween(nevName.name,1, backs-1));
        save([outputDir filename],'waveData','-v7.3')
        fileCount = fileCount +1;
        end
        waveData=[]; %memory reduction
    end
    clear waveData;
    fprintf("%d testing  file(s) saved to %s\n",fileCount,[outputDir])
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