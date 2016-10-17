function cnnPreprocess()
    trainFiles = {};
    testFiles = {};
    for n = 0:9
        train = sprintf('..\\Images_Data_Clipped\\Train\\%s\\*.jpg', int2str(n));
        test = sprintf('..\\Images_Data_Clipped\\Test\\%s\\*.jpg', int2str(n));
        trainFiles = [trainFiles, train];
        testFiles = [testFiles, test];
    end

    dataTrainstore = imageDatastore( trainFiles,'LabelSource','foldernames');
    dataTeststore = imageDatastore( testFiles,'LabelSource','foldernames');
    
    dataTrainstore = shuffle(dataTrainstore);
    dataTeststore = shuffle(dataTeststore);
    
    save('dataTrainstore.mat', 'dataTrainstore');
    save('dataTeststore.mat', 'dataTeststore');
    
end