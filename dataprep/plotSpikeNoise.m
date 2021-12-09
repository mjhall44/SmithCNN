
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

