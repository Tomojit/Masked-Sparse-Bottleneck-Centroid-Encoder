import pdb
import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from utilityScript import *
from SparseAdaptiveBottleneckCentroidencodePyTorch import SABCE
from simpleANNClassifierPyTorch import *


def load_COIL20_Data(dataSetName,partition=None):
	
	#load data file
	trnSet,tstSet = getApplicationData(dataSetName,partition)
	
	return trnSet,tstSet
	
def runSABCE(trData,trLabels,conf_dict,num_epochs_pre,num_epochs_post,learning_rate,
			gpuId,standardizeFlag):
	
	
	miniBatch_size = trData.shape[1]
	conf_dict['inputDim'] = np.shape(trData)[1]
	model = SABCE(conf_dict)
	model.fit(trData,trLabels,learning_rate,miniBatch_size,num_epochs_pre,num_epochs_post,standardizeFlag,gpuId,verbose=False)
	splWs = model.splWs.detach().to('cpu')
	feaList,feaW = returnImpFeaturesElbow(splWs)
	return feaList
	
def classifyCOIL20Data(trnSet,tstSet,featureSet,fCntList,gpuId,pp):
	
	accuracyList = []
	
	for feaCnt in fCntList:
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		tstData,tstLabels = tstSet[:,:-1],tstSet[:,-1]
		
		#use the selected features
		fea = featureSet[:feaCnt]
		trData,tstData = trData[:,fea],tstData[:,fea]	
		nClass = len(np.unique(trLabels))
		allACC = []
		for i in range(10):
			ann = NeuralNet(trData.shape[1], [1500] , nClass)
			ann.fit(trData,trLabels,standardizeFlag=True,batchSize=64,optimizationFunc='Adam',learningRate=0.001, numEpochs=200,cudaDeviceId=gpuId)
			ann = ann.to('cpu')
			tstPredProb,tstPredLabel = ann.predict(tstData)
			accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
			allACC.append(accuracy)	
		allACC = np.hstack((allACC))
		accuracyList.append(np.round(np.mean(allACC),2))
		
		print('Repetition:',pp+1,'Accuracy using',trData.shape[1],'of features:',np.round(np.mean(allACC),1))
	return accuracyList


if __name__ == "__main__":

	dataSet = 'COIL20'

	# hyper-parameters for Adam optimizer
	num_epochs_pre = 10
	num_epochs_post = 1050
	learning_rate = 0.008
	gpuId = 12
	fCntList = [50]
	standardizeFlag = True


	# initialize network hyper-parameters
	conf_dict = {}
	conf_dict['hL'] = [100]
	conf_dict['hActFunc'] = ['tanh']
	conf_dict['oActFunc'] = 'linear'
	conf_dict['errorFunc'] = 'MSE'
	conf_dict['Lambda1'] = 0.001
	conf_dict['Lambda2'] = 0.001
	conf_dict['mu1'] = 0.8
	conf_dict['mu2'] = 0.3

	topFiftyFeaturesAcc = []
	for pp in range(20):

		#load training data data
		trnSet,tstSet = load_COIL20_Data('COIL20')
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		
		#run SCE on training data for feature selection
		feaList = runSABCE(trData,trLabels,conf_dict,num_epochs_pre,num_epochs_post,learning_rate,gpuId,standardizeFlag)

		#using the selected features run classification
		accuracyList = classifyCOIL20Data(trnSet,tstSet,feaList,fCntList,gpuId,pp)
		topFiftyFeaturesAcc.append(accuracyList[0])
	print('\t Mean accuracy using top 50 features over 20 run',np.round(np.mean(topFiftyFeaturesAcc),2))

