import os
from glob import glob
from random import shuffle

path = os.getcwd()
ALLFILES = glob(path + '/dog-breed-identification/data/train/' + '*/*.jpg')

def makeNewTrainTxtFiles():

    trainfile = open(path + '/dog-breed-identification/train.txt', 'w')
    testfile = open(path + '/dog-breed-identification/test.txt', 'w')

    num_of_training_files = 0.7*len(ALLFILES)
    count =0
    for i in ALLFILES:
        count+=1
        if count<=num_of_training_files:
            trainfile.write(i + '\n')
        else:
            testfile.write(i + '\n')
    trainfile.close()
    testfile.close()

if __name__ == "__main__":
    makeNewTrainTxtFiles()