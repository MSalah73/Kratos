# -*- coding: utf-8 
import numpy as np
import cv2
import torch
import tqdm
#import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
txt_path = "list_color_cloth.txt"
dblTrain = []
dblValidation = []
#customer dataloader
class ColorCustomDataset(Dataset):
    def __init__(self, txt_path, transform):
        self.imageset = []
        self.labelset = []
        f = open("list_color_cloth.txt")
        i = 0
        while i < 52714:
            line = f.readline()
            if i != 0 and i != 1:
                img_path, color = line.split(maxsplit=1)
                self.imageset.append(img_path)
                self.labelset.append(color)
            i += 1
        f.close()
        temp = self.labelset
        self.classes = list(set(temp))#no replicates color label list, all colors the model could predict on
        #replace letter labels by int index
        temp = []
        for label in self.labelset:
            label = self.classes.index(label)
            temp.append(label)
        self.labelset = temp
        #self.classes = np.asarray(list(set(temp)), np.uint)#numpy array of all record colors
        #self.labelset = np.asarray(self.labelset, np.int32)#numpy array of lables
        self.transform = transform
        
    def __getitem__(self, index):
        img = cv2.imread(self.imageset[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#RGB numpy array
        img = cv2.resize(img, (112,112), interpolation = cv2.INTER_AREA)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labelset[index]
        return (img, label)

    def __len__(self):
        return len(self.imageset)
transform = transforms.Compose([transforms.ToTensor()])
#split dataset into train and test set and load the dataset
dataset = ColorCustomDataset(txt_path, transform)
batch_size = 64
validation_split = .2
shuffle = True
random_seed= 64

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

objectTrain = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
objectValidation = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1_bn = torch.nn.BatchNorm2d(48)
        self.conv2_bn = torch.nn.BatchNorm2d(64)
        self.conv5_bn = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(3, 48, kernel_size=11)
        self.conv2 = torch.nn.Conv2d(48, 64, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(64,192, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(192, 96, kernel_size=3)
        self.conv5 = torch.nn.Conv2d(96, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64*9*9, 4096)
        self.fc2 = torch.nn.Linear(4096, 961)
	# end
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1_bn(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2) 
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2_bn(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=1)
        #x1, x2 = torch.split(x, 96)
        #print(x1.shape)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv5_bn(x)
        x = torch.nn.functional.relu(x)
        #x = torch.flatten(x)
        x = x.view(-1, 5184)
        x = self.fc1(x)
        x = torch.nn.functional.dropout(x, p=0.35, training=self.training)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)
	# end
# end

moduleNetwork = Network()

# specifying the optimizer based on adaptive moment estimation, adam
# it will be responsible for updating the parameters of the network

objectOptimizer = torch.optim.Adam(params=moduleNetwork.parameters(), lr=0.001)

def train():
	# setting the network to the training mode, some modules behave differently during training

	moduleNetwork.train()

	# obtain samples and their ground truth from the training dataset, one minibatch at a time

	for tensorInput, tensorTarget in tqdm.tqdm(objectTrain):
		# wrapping the loaded tensors into variables, allowing them to have gradients
		# in the future, pytorch will combine tensors and variables into one type
		# the variables are set to be not volatile such that they retain their history

		variableInput = torch.autograd.Variable(data=tensorInput, volatile=False)
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=False)

		# setting all previously computed gradients to zero, we will compute new ones

		objectOptimizer.zero_grad()

		# performing a forward pass through the network while retaining a computational graph

		variableEstimate = moduleNetwork(variableInput)

		# computing the loss according to the cross-entropy / negative log likelihood
		# the backprop is done in the subsequent step such that multiple losses can be combined

		variableLoss = torch.nn.functional.nll_loss(input=variableEstimate, target=variableTarget)

		variableLoss.backward()

		# calling the optimizer, allowing it to update the weights according to the gradients

		objectOptimizer.step()
	# end
# end

def evaluate():
	# setting the network to the evaluation mode, some modules behave differently during evaluation

	moduleNetwork.eval()

	# defining two variables that will count the number of correct classifications

	intTrain = 0
	intValidation = 0

	# iterating over the training and the validation dataset to determine the accuracy
	# this is typically done one a subset of the samples in each set, unlike here
	# otherwise the time to evaluate the model would unnecessarily take too much time

	for tensorInput, tensorTarget in objectTrain:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True)
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)

		variableEstimate = moduleNetwork(variableInput)

		intTrain += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()
	# end

	for tensorInput, tensorTarget in objectValidation:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True)
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)

		variableEstimate = moduleNetwork(variableInput)

		intValidation += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()
	# end

	# determining the accuracy based on the number of correct predictions and the size of the dataset

	dblTrain.append(100.0 * intTrain / len(objectTrain.dataset))
	dblValidation.append(100.0 * intValidation / len(objectValidation.dataset))

	print('')
	print('train: ' + str(intTrain) + '/' + str(len(objectTrain.dataset)) + ' (' + str(dblTrain[-1]) + '%)')
	print('validation: ' + str(intValidation) + '/' + str(len(objectValidation.dataset)) + ' (' + str(dblValidation[-1]) + '%)')
	print('')
# end

# training the model for 100 epochs, one would typically save / checkpoint the model after each one

for intEpoch in range(100):
	train()
	evaluate()
