from lib import *
from dataAnalysis import analysisData
from dataPreprocessing import preprocessData
from model import getTrainData, buildModel
from test import *

dataSet = pd.read_csv("datasets/clinical_record.csv")

# print("\n")
# print(dataSet.head())

# print("\n")
# dataSet.info()
# print("\n")
# print(dataSet.isnull().sum())




# analysisData(dataSet)

preprocessData(dataSet)
trainData = getTrainData(dataSet)
buildModel(trainData)
# testannModel(trainData)
