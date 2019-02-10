import math
from struct import unpack
import gzip
import numpy as np
from numpy import zeros, uint8, float32
import random
import matplotlib.pyplot as plt

class Perceptron:
	def __init__(self, numInputs):
		#contains weights, bias, sum, value
		
		#weights contains two items, first is the weight vector, second is the weight grad. descent
		w1 = np.random.rand(numInputs) - np.tile(0.5, (numInputs))
		w2 = np.zeros(numInputs)
		self.weights = [w1, w2]
		
		#bias contains two items, first is the bias term, second is the bias grad. descent
		self.bias = []
		
		self.bias.append(random.random())
		self.bias.append(0)
		
		#sum is weights * previous layer plus the bias term, value is the sig(sum)
		self.sum = 0
		self.value = 0
		
	def set_weights(self, w):
		#input must be a vector of two vectors
		self.weights = w
		
	def get_weights(self):
		return self.weights
	
	def set_bias(self, b):
		#input must be a vector of two values
		self.bias = b
		
	def get_bias(self):
		return self.bias
		
	def set_sum(self, s):
		self.sum = s
	
	def get_sum(self):
		return self.sum
	
	def set_value(self, v):
		self.value = v
		
	def get_value(self):
		return self.value
		
	def make_weights_adjustments(self):
		self.weights[0] -= self.weights[1]
		
	def make_bias_adjustments(self):
		self.bias[0] -= self.bias[1]
		
	def getsigprimevalue (self):
		if self.sum * -1 > 300:
			return 0
		if self.sum * -1 < -300:
			return 1
		return 1 / math.pow((1 + math.pow(math.e, self.sum * -1)), 2)
	
	def calculate_sum(self, previous):
		previous_layer_values = np.array(previous)
		self.sum = np.dot(previous_layer_values, self.weights[0]) + self.bias[0]
		
		
	def calculate_value(self, previous):
		self.calculate_sum(previous)
		
		self.sum += self.bias[0]
		
		#print(self.sum)
		if self.sum < -500:
			self.value = 0
		elif self.sum > 500:
			self.value = 1
		else:
			self.value = self.sigmoid(self.sum)
		
	def sigmoid(self, x):
		return 1 / (1 + math.pow(math.e, x * -1))



class Neural_Networks:
	def __init__(self, numInputs, numHidden, layersHidden, numImagesCompiled):
		
		self.numInputs = numInputs
		self.numHidden = numHidden
		self.layersHidden = layersHidden

		
		print("Number of Inputs: " + str(numInputs) + "\n")
		print("Number of Hidden Layers: " + str(numHidden) + "\n")
		for i in range(numHidden):
			print("Number of Perceptrons in Layer " + str(i+1) + ": " + str(layersHidden[i]) + "\n")
		
		self.perceptron_net = []
		#Hidden Layer
		for layer in range(self.numHidden):
			temp = []
			if layer == 0:
				for j in range(self.layersHidden[i]):
					temp.append(Perceptron(self.numInputs))
			else:
				for j in range(self.layersHidden[i]):
					temp.append(Perceptron(self.layersHidden[i-1]))
			self.perceptron_net.append(temp)
			
		#Results Layer
		temp = []
		for item in range(10):
			temp.append(Perceptron(self.layersHidden[i]))
		self.perceptron_net.append(temp)
		
	def setup(self, numImagesCompiled):

			import gzip
			images = gzip.open('train-images-idx3-ubyte.gz','rb')
			labels = gzip.open('train-labels-idx1-ubyte.gz', 'rb')
			
			images.read(4)
			number_of_images = images.read(4)
			number_of_images = unpack('>I', number_of_images)[0]
			rows = images.read(4)
			rows = unpack('>I', rows)[0]
			cols = images.read(4)
			cols = unpack('>I', cols)[0]
			
			labels.read(4)  # skip the magic_number
			N = labels.read(4)
			N = unpack('>I', N)[0]
				
			if number_of_images != N:
				raise Exception('number of labels did not match the number of images')
				
			# Get the data
			x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
			y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
			for i in range(numImagesCompiled):
				if i % numImagesCompiled == 0:
					print("i: %i" % i)
				for row in range(rows):
					for col in range(cols):
						tmp_pixel = images.read(1)  # Just a single byte
						tmp_pixel = unpack('>B', tmp_pixel)[0]
						x[i][row][col] = tmp_pixel
				tmp_label = labels.read(1)
				y[i] = unpack('>B', tmp_label)[0]
			return (x, y)
			
	def print_net(self):
		end = False
		count = 0
		while not end == True:
			written = False
			if count == 0:
				for i in range(len(self.perceptron_net)):
					print("     Layer %-1d" % (i + 1), end="")
				print("")
				
			for i in range(len(self.perceptron_net)):
				if(count < len(self.perceptron_net[i])):
					print("%12.3f" % self.perceptron_net[i][count].get_value(), end="")
					written = True
				else:
					print("%12.3s" % " ", end="")
			if count >= 0 and count < 10:
				print("  [%d]" % count, end="")
			print(" ")
			
			if written == False:
				end = True
			
			count += 1
	def get_perceptron_net(self):
		return self.perceptron_net
	
	def get_value_vector_layer(self, layer):
		answer = []
		for i in self.perceptron_net[layer]:
			answer.append(i.get_value())
		return answer
		
	def calculate_result(self, image):
		imageArray = image.ravel()
		
		for layer in range(len(self.perceptron_net)):
			for percept in self.perceptron_net[layer]:
				if layer == 0:
					percept.calculate_value(imageArray);
				else:
					percept.calculate_value(self.get_value_vector_layer(layer - 1))
	
	def gradient_descent(self, image, label, lr):
		imageArray = image.ravel()
		for i in range(len(self.perceptron_net)):
			layer = len(self.perceptron_net) - 1 - i
			for perceptNum in range(len(self.perceptron_net[layer])):
				
				gradient = 0
				if layer == len(self.perceptron_net) - 1:
					if perceptNum == label[0]:
						gradient = 2 * (self.perceptron_net[layer][perceptNum].get_value() - 1)
					else:
						gradient = 2 * (self.perceptron_net[layer][perceptNum].get_value() - 0)
				else:
					for next_layer_percept_num in range(len(self.perceptron_net[layer + 1])):
						gradient += self.perceptron_net[layer + 1][next_layer_percept_num].get_weights()[1][perceptNum] / self.perceptron_net[layer + 1][next_layer_percept_num].get_value() * self.perceptron_net[layer + 1][next_layer_percept_num].get_weights()[0][next_layer_percept_num]
				gradient *= self.perceptron_net[layer][perceptNum].getsigprimevalue()
				
				#setting bias adj.
				old_bias = self.perceptron_net[layer][perceptNum].get_bias()
				new_bias = old_bias[1] + lr * gradient
				self.perceptron_net[layer][perceptNum].set_bias([old_bias[0], new_bias])
				
				#setting weights adj.
				old_weights = self.perceptron_net[layer][perceptNum].get_weights()
				new_weights_adj = old_weights[1]
				
				
				for i in range(len(new_weights_adj)):
					if not layer == 0:
						new_weights_adj[i] += self.perceptron_net[layer - 1][i].get_value() * lr * gradient
					else:
						new_weights_adj[i] += imageArray[i] * lr * gradient
			
				self.perceptron_net[layer][perceptNum].set_weights([old_weights[0], new_weights_adj])
	def make_adjustments(self):
		for i in self.perceptron_net:
			for j in i:
				j.make_weights_adjustments()
				j.make_bias_adjustments()

def train_network(network, learning_rate, images, labels, epoch, total):	
	for i in range(total):
		network.calculate_result(images[i])
		n=network.gradient_descent(images[i], labels[i], learning_rate)
		if i % epoch == 0:
			print(i)
			network.make_adjustments()
			
def test(network, number):
	network.calculate_result(images[number])
	network.print_net()	
	print(labels[number])
	

n = Neural_Networks(28*28, 2, [16, 16], 1000)
(images, labels) = n.setup(1000)

#n.calculate_result(images[0])
#n.print_net()
#n.gradient_descent(images[0], labels[0], 1)
#n.make_adjustments()
#n.calculate_result(images[0])
#n.print_net()
test(n, 0)
train_network(n, 0.5, images, labels, 10, 100)
test(n, 0)
#test(n, 1)
#test(n, 2)
#test(n, 3)


	
	



