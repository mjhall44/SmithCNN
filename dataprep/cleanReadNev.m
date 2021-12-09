function waveData = cleanReadNev(fullPath, chan)
% Uses the read_nev function, while converting to binary labels on read

    [spikes,waves] = read_nev(fullPath,'channels',chan);
    waves = [waves{:}]';
    waveData = [nan(size(waves,1),1),waves];
    waveData(:,1) = spikes(:,2);

    fprintf('unq before: ')
    unq =unique(waveData(:,1));
    for k1 = 1:numel(unq)
       fprintf('%d ', unq(k1))
    end
    fprintf('\n')
    waveData(logical(waveData(:,1)==255),1) = 0;
    waveData(logical(waveData(:,1)> 1),1) = 1;
    fprintf('unq after : ')
    unq =unique(waveData(:,1));
    for k1 = 1:numel(unq)
       fprintf('%d ', unq(k1))
    end
    fprintf('\n')
end
