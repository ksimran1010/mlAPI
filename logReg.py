import pandas as pd
import numpy as np


# filePath = "C:\\Users\\karsi\\PycharmProjects\\the proect\\regresseion\\regdata.csv"


class logistic():
    def __init__(self, iterations):
        self.path = "data/regdata.csv"
        self.iter = iterations

    def sig(self, x, theta):
        z = np.dot(x, theta.T)
        return 1 / (1 + np.exp(-z))

    def costfxn(self, x, y, theta):
        cc = -y * (np.log(self.sig(x, theta))) - (1 - y) * (np.log(1 - self.sig(x, theta)))
        return np.mean(cc)

    def logReg(self, xval, lbl, iterations, l=0.1):
        beta = np.zeros(xval.shape[1])
        for i in range(0, iterations):
            beta = beta - ((l / xval.shape[0]) * np.dot(xval.T, self.sig(xval, beta) - lbl))
        return beta

    def readdata(self):
        data = pd.read_csv(self.path)
        data = np.array(data)
        from sklearn.model_selection import train_test_split
        dataall = data[:, 1:3]
        datalbl = data[:, 3]
        return train_test_split(dataall, datalbl, test_size=0.5, stratify=datalbl)

    def run(self):
        trainDataX, testDataX, trainDataY, testDataY = self.readdata()
        theta = self.logReg(trainDataX, trainDataY, self.iter)
        yPredicted = np.round(self.sig(testDataX, theta))
        accuracy = np.sum((yPredicted - testDataY) == 0) / len(yPredicted)
        return accuracy

#logObj = logistic(100)
#acc = logObj.run()
#print('final accuracy by logistic regression is ',acc)
