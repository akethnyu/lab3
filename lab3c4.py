import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import time
import torch.distributed as dist
from random import Random

dist.init_process_group(backend='mpi', world_size=5)
rank = dist.get_rank()
wsize = dist.get_world_size()
seed = 123
torch.manual_seed(seed)
paraserv_rank = wsize - 1
#dataloader_io = 0
#dataloader_preprocessing = 0
#loadbatch_time = 0
#epoch_time = 0

class LandmarksDataset(Dataset):
   def __init__(self,csv_file, root_dir,Transform=None):

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.image_id = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = Transform

   def __len__(self):
        return len(self.image_id)

   def __getitem__(self, idx):

        global io_time
        global augmentation_time

        img_name = os.path.join(self.root_dir,self.image_id.iloc[idx, 0])

        #start = time.monotonic()

        image = Image.open(img_name + '.jpg')
        label = self.image_id.iloc[idx,1]

        #end = time.monotonic()

        #io_time += (end - start)

        #start = time.monotonic()

        rgbimage = image.convert('RGB')
        sample = {'image': rgbimage, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        #end = time.monotonic()

        #augmentation_time += (end - start)

        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 17)


    def dummy_backprop(self):
        inputs = Variable(torch.randn(1,32 * 32 * 3,requires_grad=True))
        outputs = self(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = (outputs[0].mean()) * 0.0
        loss.backward()
        #loss = criterion(outputs, labels)
        #loss.backward()

    def forward(self,x):
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=0)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Partition(object):
	def __init__(self, data, index):
	 	self.data = data
	 	self.index = index

	def __len__(self):
	 	return len(self.index)
	
	def __getitem__(self, index):
	 	data_idx = self.index[index]
	 	return self.data[data_idx]

class DataPartitioner(object):
	def __init__(self, data, sizes , seed=1234):
		self.data = data
		self.partitions = []
		rng = Random()
		rng.seed(seed)
		data_len = len(data)
		indexes = [x for x in range(0, data_len)]
		rng.shuffle(indexes)
		
		for frac in sizes:
			part_len = int(frac * data_len)
			self.partitions.append(indexes[0:part_len])
			indexes = indexes[part_len:]

	def use(self, partition):
		return Partition(self.data, self.partitions[partition])


""" Partitioning Kaggle Amazon """
def partition_dataset(dataset):
	size = dist.get_world_size() 
	bsz = 100
	partition_sizes = [1.0 / size for _ in range(size)]
	partition = DataPartitioner(dataset, partition_sizes)
	partition = partition.use(rank)
	train_set = torch.utils.data.DataLoader(partition,
	batch_size=bsz,
	shuffle=True)

	return train_set, bsz

def apply_gradients(net,optimizer):
	
	if(rank != paraserv_rank):
		for param in net.parameters():
			#print("sending from ", rank)
			dist.send(tensor=param.grad.data, dst=paraserv_rank)

		for param in net.parameters():
                        #print("recieving from ", rank)
                        dist.recv(tensor=param, src=paraserv_rank)

	if(rank == paraserv_rank):
		for i in range(wsize - 1):
			sender = None
			for param in net.parameters():
				#print("recieving gradients:")
				if(sender == None):
					sender = dist.recv(tensor=param.grad.data)
				else:
					dist.recv(tensor=param.grad.data,src=sender)

			optimizer.step()

			for param in net.parameters():
				#print("sending parameters")
				dist.send(tensor=param, dst=sender)
		
	

def run():

	#if(rank != paraserv_rank):

	train_dataset = LandmarksDataset(csv_file='/home/atm423/CSCI-GA.3033-023/kaggleamazon/train.csv',
			root_dir='/home/atm423/CSCI-GA.3033-023/kaggleamazon/train-jpg',
			Transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

	train_load, dw = partition_dataset(train_dataset)

	model = Net()
	model.dummy_backprop()
	model.zero_grad()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
	running_loss = 0.0
	num_minibatch = 0

	tstart = time.monotonic()
	for epoch in range(5):
			for i,data in enumerate(train_load,0):
			
				inputs,labels = data["image"],data["label"]
				inputs = Variable(inputs)
				labels = Variable(labels)

				optimizer.zero_grad()
				output = model(inputs)
				loss = criterion(output, labels)
				loss.backward()

				if(paraserv_rank == rank): 
					optimizer.step()

				apply_gradients(model,optimizer)
				running_loss += loss.item()
				num_minibatch += 1
				
			torch.distributed.barrier()
	

			if(rank == paraserv_rank): 
				running_loss /= num_minibatch
				print(running_loss)

	tend = time.monotonic()

	epoch_time = tend - tstart

	dwlw = torch.zeros(1)
	dwlw[0] = dw * running_loss
	dist.all_reduce(dwlw, op=dist.reduce_op.SUM)
	dwlw[0] /= len(train_dataset)

	print('finished training')
	print("C4 loss : %f, avg epoch_time : %f\n" % (dwlw[0], epoch_time / 5))

if __name__ == "__main__":
    run()
