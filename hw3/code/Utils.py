from datetime import datetime

def startTime():
    date_time = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    print("================================================")
    print("Start Time: ", date_time)
    print("------------------------------------------------")

def endTime():
    date_time = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    print("------------------------------------------------")
    print("End Time: ", date_time)
    print("================================================")

def readFileToArray(path):
    file = open(path, "r")
    result = list(map(lambda x : x.strip(), file.readlines()))
    file.close()
    return result

stop = readFileToArray("data/stoplist.txt")
xTrain = readFileToArray("data/traindata.txt")
yTrain = readFileToArray("data/trainlabels.txt")
xTest = readFileToArray("data/testdata.txt")
yTest = readFileToArray("data/testlabels.txt")