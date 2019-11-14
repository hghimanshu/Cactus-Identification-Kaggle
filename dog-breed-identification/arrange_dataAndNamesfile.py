import os


def makeDataFile():
    data_file = path + '/dog-breed-identification/classifier.data'
    with open(data_file,'w+') as f:
        f.write('classes='+str(len(total_classes))+'\n')
        f.write('train='+trainfile+'\n')
        f.write('valid='+testfile+'\n')
        f.write('names='+namesfile+'\n')
        f.write('backup='+backupfile+'\n')



def makeNameFile():
    with open(names_path + 'classifier.names', 'w') as f:
        for name in total_classes:
            f.write(name + '\n')

if __name__ == "__main__":
    path = os.getcwd()
    data_folder = path + '/dog-breed-identification/data/train/'
    names_path = path + '/dog-breed-identification/'
    trainfile = path + '/dog-breed-identification/train.txt'
    testfile = path + '/dog-breed-identification/test.txt'
    namesfile = path + '/dog-breed-identification/classifier.names'
    backupfile = '/home/himanshu/Himanshu/alpr-unconstrained/darknet/backup/'
    total_classes = os.listdir(data_folder)


    makeNameFile()
    makeDataFile()
    