import os


def makeDataFile(trainfile, testfile, totalimages):
    pass


def makeNameFile(clasfolder, names_path):
    total_classes = os.listdir(clasfolder)
    with open(names_path + 'classifier.names', 'w') as f:
        for name in total_classes:
            f.write(name + '\n')

if __name__ == "__main__":
    path = os.getcwd()
    data_folder = path + '/dog-breed-identification/data/train/'
    names_path = path + '/dog-breed-identification/'
    makeNameFile(data_folder, names_path)