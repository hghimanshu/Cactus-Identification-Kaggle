import os


def makeDataFile(trainfile, testfile, totalimages):
    pass


def makeNameFile(clasfolder):
    total_classes = os.listdir(clasfolder)
    with open('classifier.names', 'w') as f:
        for name in total_classes:
            f.write(name + '\n')

if __name__ == "__main__":
    path = os.getcwd()
    print(path)