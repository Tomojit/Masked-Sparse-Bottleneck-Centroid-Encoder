import pdb
from copy import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

class SABCE(nn.Module):
	def __init__(self, netConfig={}):
		super(SABCE, self).__init__()
		if len(netConfig.keys())!=0:		
			self.inputDim,self.outputDim = netConfig['inputDim'],netConfig['inputDim']
			self.hLayer,self.hLayerPost = copy(netConfig['hL']),copy(netConfig['hL'])
			for i in range(len(self.hLayerPost)-1,0,-1):
				self.hLayerPost.extend([self.hLayerPost[i-1]])

			self.Lambda1,self.Lambda2 = 0.001,0.001
			self.oActFunc,self.errorFunc = 'linear','MSE'
			
			#pdb.set_trace()
			if 'Lambda1' in netConfig.keys(): self.Lambda1 = netConfig['Lambda1']
			if 'Lambda2' in netConfig.keys(): self.Lambda2 = netConfig['Lambda2']
			
			if 'mu1' in netConfig.keys(): self.mu1 = netConfig['mu1']
			if 'mu2' in netConfig.keys(): self.mu2 = netConfig['mu2']
			
			if 'errorFunc' in netConfig.keys(): self.errorFunc = netConfig['errorFunc']
			if 'oActFunc' in netConfig.keys(): self.oActFunc = netConfig['oActFunc']

			self.hActFunc,self.hActFuncPost = copy(netConfig['hActFunc']),copy(netConfig['hActFunc'])
			for i in range(len(self.hActFuncPost)-1,0,-1):
				self.hActFuncPost.extend([self.hActFuncPost[i-1]])

		else:#for default set up
			self.hLayer = [100]
			self.oActFunc,self.errorFunc = 'linear','MSE'
			self.hActFunc,self.hActFuncPost = 'tanh','tanh'
			self.mu1 = 0.5
			self.mu2 = 0.25

		self.device = None
		#internal variables
		self.epochError = []
		self.epochErrorPre = []
		self.trMu = []
		self.trSd = []
		self.tmpPreHActFunc = []
		self.optFeaCnt = 0


	def initNet(self,input_size,hidden_layer):
		self.hidden=nn.ModuleList()
		# Hidden layers
		if len(hidden_layer)==1:
			self.hidden.append(nn.Linear(input_size,hidden_layer[0]))
		elif(len(hidden_layer)>1):
			for i in range(len(hidden_layer)-1):
				if i==0:
					self.hidden.append(nn.Linear(input_size, hidden_layer[i]))
					self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
				else:
					self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		self.reset_parameters(hidden_layer)
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], input_size)
		#sparse layer
		self.splWs = nn.Parameter(torch.ones(self.inputDim,))

	def reset_parameters(self,hidden_layer):
		#pdb.set_trace()
		tmpActFunc = self.hActFunc[:int(np.ceil(len(hidden_layer)/2))]
		for i in range(len(tmpActFunc)-1,0,-1):
			tmpActFunc.extend([tmpActFunc[i-1]])
		hL = 0
		
		while True:
			#pdb.set_trace()
			if tmpActFunc[hL].upper() in ['SIGMOID','TANH']:
				#pdb.set_trace()
				torch.nn.init.xavier_uniform_(self.hidden[hL].weight)
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
				#continue
			elif tmpActFunc[hL].upper() == 'RELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif tmpActFunc[hL].upper() == 'LRELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='leaky_relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			if hL == len(hidden_layer)-1:
				break
			hL += 1

	def forwardPost(self, x):

		#forward pass using encoder and decoder on the input data
		return self.decoder(self.encoder(x))
			
	def encoder(self,x):
		#forward pass for encoder on the inout data
		D = torch.diag(self.splWs)
		x = torch.matmul(x,D)
		
		for l in range(len(self.hActFunc)):
			if self.hActFunc[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear				
				x = self.hidden[l](x)
		return x
		
	def decoder(self,x):
		#forward pass for decoder on the encoder data
		#pdb.set_trace()
		for l in range(len(self.hActFunc),len(self.hActFuncPost)):
			if self.hActFuncPost[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear				
				x = self.hidden[l](x)
		#for putput layer
		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)
		return x

	def forwardPre(self, x):
		# Feedforward
		D = torch.diag(self.splWs)
		x = torch.matmul(x,D)
		for l in range(len(self.hidden)):
			if self.tmpPreHActFunc[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear
				x = self.hidden[l](x)

		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)
		
	def createCentroidTarget(self,data,label):
		#pdb.set_trace()
		centroidLabels=np.unique(label)
		outputData=np.zeros([np.shape(data)[0],np.shape(data)[1]])
		for i in range(len(centroidLabels)):
			indices=np.where(centroidLabels[i]==label)[0]
			tmpData=data[indices,:]
			centroid=np.mean(tmpData,axis=0)
			outputData[indices,]=centroid
		return outputData
		
	def createEncoderOutputAsCentroids(self,D,L):
		#D: an [n x d] matrix
		#	n: no of samples
		#	d: dimension of data
		outputData = torch.zeros([D.shape[0],D.shape[1]],device=self.device)
		nClass = len(torch.unique(L))
		#with torch.no_grad():#I don't need to calculate gradient for this operations
		for i in range(nClass):
			indices = torch.where(L==i)[0]
			tmpData = D[indices,:]
			centroid = torch.mean(tmpData,axis=0)
			outputData[indices,] = centroid
		#pdb.set_trace()
		return outputData
		
	def calcInterClassDistances(self,D,L):
		#D: an [n x d] matrix
		#       n: no of samples
		#       d: dimension of data
		#This function will calculate the distance from the centroid of each class pairs in bottleneck
		#Take the logarithm of the distances and maximises the sum of the logs.
		
		classes = torch.unique(L)
		bottleneckC = torch.zeros([len(classes),D.shape[1]],device=self.device)
		for c in range(len(classes)):
			indices = torch.where(L==classes[c])[0]
			centroid = torch.mean(D[indices,:],axis=0)
			bottleneckC[c,:] = centroid
		separationLoss = 0
		#pdb.set_trace()
		for i in range(len(bottleneckC)-1):
			for j in range(i+1,len(bottleneckC)):
				separationLoss += 1/(1+torch.norm((bottleneckC[i]-bottleneckC[j]),dim=0))
		
		return separationLoss
		
	def standardizeData(self,data,mu=[],std=[]):
		#data: a m x n matrix where m is the no of observations and n is no of features
		if not(len(mu) and len(std)):
			#pdb.set_trace()
			std = np.std(data,axis=0)
			mu = np.mean(data,axis=0)
			std[np.where(std==0)[0]] = 1.0 #This is for the constant features.
			standardizeData = (data - mu)/std
			return mu,std,standardizeData
		else:
			standardizeData = (data - mu)/std
			return standardizeData
			
	def unStandardizeData(self,data,mu,std):
		return std * data + mu
		

	def preTrain(self,dataLoader,learningRate,batchSize,numEpochs,verbose):
		
		#loop to do layer-wise pre-training
		for d in range(len(self.hLayer)):
			
			#set the hidden layer structure for a bottleneck architecture
			hidden_layer=self.hLayer[:d+1]
			self.tmpPreHActFunc=self.hActFunc[:d+1]
			for i in range(len(hidden_layer)-1,0,-1):
				hidden_layer.extend([hidden_layer[i-1]])
				self.tmpPreHActFunc.extend([self.tmpPreHActFunc[i-1]])

			if verbose:
				if d==0:
					print('Pre-training layer [',self.inputDim,'-->',hidden_layer[0],'-->',self.inputDim,']')
				else:
					index=int(len(hidden_layer)/2)
					print('Pre-training layer [',hidden_layer[index-1],'-->',hidden_layer[index],'-->',hidden_layer[index+1],']')			

			#initialize the network weight and bias
			self.initNet(self.inputDim,hidden_layer)

			#freeze pretrained layers
			if d>0:
				j=0#index for preW and preB
				for l in range(len(hidden_layer)):
					if (l==d) or (l==(d+1)):
						continue
					else:
						self.hidden[l].weight=preW[j]
						self.hidden[l].weight.requires_grad=False
						self.hidden[l].bias=preB[j]
						self.hidden[l].bias.requires_grad=False
						j+=1
				self.out.weight=preW[-1]
				self.out.weight.requires_grad=False
				self.out.bias=preB[-1]
				self.out.bias.requires_grad=False

			# set loss function
			criterion = nn.MSELoss()

			# set optimization function
			optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)

			# Load the model to device
			self.to(self.device)

			# Start training
			for epoch in range(numEpochs):
				error=[]
				for i, (trInput, trOutput,L) in enumerate(dataLoader):  
					# Move tensors to the configured device
					trInput = trInput.to(self.device)
					trOutput = trOutput.to(self.device)

					# Forward pass
					outputs = self.forwardPre(trInput)
					loss = criterion(outputs, trOutput)
					
					error.append(loss.item())

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				self.epochErrorPre.append(np.mean(error))
				if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
					print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs, self.epochErrorPre[-1]))
			
			#variable to store pre-trained weight and bias
			if d <len(self.hLayer)-1:
				preW=[]
				preB=[]
				for h in range(len(hidden_layer)):
					preW.append(self.hidden[h].weight)
					preB.append(self.hidden[h].bias)
				preW.append(self.out.weight)
				preB.append(self.out.bias)

		#now set requires_grad =True for all the layers
		for l in range(len(hidden_layer)):			
			self.hidden[l].weight.requires_grad=True			
			self.hidden[l].bias.requires_grad=True
			
		self.out.weight.requires_grad=True
		self.out.bias.requires_grad=True
		
		if verbose:
			print('Pre-training is done.')

	def postTrain(self,dataLoader,learningRate,batchSize,numEpochs,verbose):
		
		# set loss function
		criterion = nn.MSELoss()

		# set optimization function
		optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)

		# Load the model to device
		self.to(self.device)
		
		# Start training
		if verbose:
			print('Training network:',self.inputDim,'-->',self.hLayerPost,'-->',self.inputDim)
		for epoch in range(numEpochs):
			error=[]
			for i, (trInput, trOutput,L) in enumerate(dataLoader):  
				
				# Move tensors to the configured device
				trInput = trInput.to(self.device)
				trOutput = trOutput.to(self.device)
				
				#Update the class centroids using weight of sparse layer.
				#In this approach I will change the class centroids by element-wise multiplying the centroids with splWs.
				if epoch >= 1:
					#print('Norm of trOutput:',torch.norm(trOutput).item())
					#pdb.set_trace()
					with torch.no_grad():#we don't need to compute gradients for this operation
						D = torch.diag(self.splWs)
						trOutput = torch.matmul(trOutput,D)
					#print('Norm of updated trOutput:',torch.norm(trOutput).item())
				
				L = L.to(self.device)

				#pdb.set_trace()
				# Forward pass
				outputs = self.forwardPost(trInput)
				loss = criterion(outputs, trOutput)
				
				#forward pass with the output data to get a low dim embedding of centroids
				ceOutput_en = self.encoder(trOutput)
				output_en = self.encoder(trInput)
				
				#calculate CE loss with bottleneck output
				ceLoss = criterion(output_en,ceOutput_en)
				
				#calculate class separation loss with bottleneck output
				#if there is only one class in the minibatch then we don't want to calculate class seperation loss
				#this check confirms that
				if len(torch.unique(L)) > 1:
					separationLoss = self.calcInterClassDistances(ceOutput_en,L)

				#L2,1 regularization loss on SPL layer
				sparsityLoss = self.Lambda1*torch.norm(self.splWs, p=1) + self.Lambda2*torch.norm(self.splWs, p=2)
				
				#pdb.set_trace()
				if len(torch.unique(L)) > 1:#make sure separation loss has been calculated
					loss = loss + self.mu1*ceLoss + self.mu2*separationLoss + sparsityLoss
				else:
					loss = loss + self.mu1*ceLoss
				
				error.append(loss.item())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			self.epochError.append(np.mean(error))
			if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs, self.epochError[-1]))


	def fit(self,trData,trLabels,learningRate=0.001,miniBatchSize=100,numEpochsPreTrn=25,numEpochsPostTrn=100,standardizeFlag=True,cudaDeviceId=0,verbose=True):

		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))

		if standardizeFlag:
		#standardize data
			mu,sd,trData = self.standardizeData(trData)
			self.trMu = mu
			self.trSd = sd

		#create target: centroid for each class
		target = self.createCentroidTarget(trData,trLabels)

		#prepare data for Torch
		trDataTorch=Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(target).float(),torch.from_numpy(trLabels))
		dataLoader=Data.DataLoader(dataset=trDataTorch,batch_size=miniBatchSize,shuffle=True)

		#layer-wise pre-training		
		self.preTrain(dataLoader,learningRate,miniBatchSize,numEpochsPreTrn,verbose)
		if verbose:
			print('Norm of SPL layer after pre-training',torch.norm(self.splWs))
		#post training
		self.postTrain(dataLoader,learningRate,miniBatchSize,numEpochsPostTrn,verbose)
		if verbose:
			print('Norm of SPL layer after post-training',torch.norm(self.splWs))
		
	def predict(self,x):
		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = self.standardizeData(x,self.trMu,self.trSd)
		#pdb.set_trace()
		x=torch.from_numpy(x).float().to(self.device)
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			fOut = self.encoder(x)
		return fOut
