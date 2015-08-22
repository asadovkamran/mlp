# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import interface as int
import readme
import asthma as a

def loadDataSet(filename, delimiter):
	
	if filename == 'asthma.txt':
		data = a.getAsthma(filename)
	else:
		data = sp.genfromtxt(filename, delimiter=delimiter)
	
	x = data[:, 0:-1] #rows without the last element
	y = data[:,-1]    #rows only with the last element
	
	return x, y


def create_matrix(a, b):
	m = np.zeros((a,b))
	return m
	
def fill_matrix(matrix, wa, wb):
	for line in range(np.shape(matrix)[0]):
		for element in range(np.shape(matrix)[1]):
			matrix[line, element] = np.random.uniform(wa, wb)
	return matrix	
	
def init_mlp(p):
	il_size = p['input layer size']
	hl_size = p['hidden layer size']
	ol_size = p['output layer size']
	wa = p['wa']
	wb = p['wb']
	
	hl_m = create_matrix(hl_size, il_size)
	hl_w = fill_matrix(hl_m, wa, wb)
	
	ol_m = create_matrix(ol_size, hl_size)
	ol_w = fill_matrix(ol_m, wa, wb)

	print "\nHidden layer weights:"
	print hl_w
	print"\nOutput layer weights:"
	print ol_w
	print """
		Net has 1 hidden layer.
		Number of neurons in input layer is %d. 
		Number of neurons in hidden layer is %d.
		Number of neurons in output layer is %d.
		Range of synaptic weights is [%f, %f].
		""" % (il_size, hl_size, ol_size, wa, wb)
	
	w = []
	w.append(hl_w)
	w.append(ol_w)
	return w

#sigmoid
def sigmoid(s):
	res = 1/(1+np.exp(-1*(s)))
	return res

#sigmoid derivative	
def dsigmoid(y):
	return sigmoid(y)*(1-sigmoid(y))

def run(vector, weights):
	hw = weights[0]
	ow = weights[1]
	
	h_out = []
	for i in range(np.shape(hw)[0]):#hidden layer size
		sum = 0
		for j in range(np.shape(hw)[1]):#input layer size
			sum += (hw[i,j]*vector[j])
		sum = sigmoid(sum)
		
		h_out.append(sum)
	
	sum = 0
	for i in range(np.shape(ow)[0]):#output layer size
		for j in range(np.shape(ow)[1]):#hidden layer size
			sum += (ow[i,j]*h_out[j])
		sum = sigmoid(sum + 1) #если выходных нейронов больше, то понадобится список
		out = sum
	
	nout = []
	nout.append(out)
	nout.append(h_out)
	
	return nout
	
def backpropagate(vector, target, net_output, weights, learn_parameters):

	net_out = net_output[0]
	hidden_output = net_output[1]
	des_response = target
	
	hw = weights[0]
	ow = weights[1]
	
	size_hw = np.shape(hw)
	size_ow = np.shape(ow)
	
	learning_speed = learn_parameters['learning speed']


	hidden_error = 0.0
	error = 0.0
	error = 0.5 * (des_response - net_out)**2
	
	hidden_error = [0.0] * size_hw[0]
	hidden_gradient = [0.0] * size_hw[0]
	delta_output = [0.0]* size_ow[1]
	delta_hidden = np.zeros((size_hw[0], size_hw[1]))
	
	output_gradient = net_out*(1 - net_out)*(des_response - net_out)

	for i in range(size_ow[0]):#output layer size
		for j in range(size_ow[1]):#hidden layer size
			delta_output[j] = output_gradient * hidden_output[j]

	for i in range(size_ow[0]):#output layer size
		for j in range(size_ow[1]):#hidden layer size
			ow[i,j] = ow[i,j] + delta_output[j] * learning_speed
	
	for j in range(size_ow[1]):#hidden layer size
		for i in range(size_ow[0]):#output layer size
			hidden_error[j] += (output_gradient * ow[i,j])
			
	for i in range(size_hw[0]):#hidden layer size
		hidden_gradient[i] = hidden_output[i]*(1-hidden_output[i])*hidden_error[i] #Oj(1-Oj)*SUM{output_gradient*ow[i,j]}
			
	for i in range(size_hw[0]):#hidden layer size
		for j in range(size_hw[1]):#input layer size
			delta_hidden[i,j] = hidden_gradient[i]*vector[j]
	
	for i in range(size_hw[0]):#hidden layer size
		for j in range(size_hw[1]):#input layer size
			hw[i,j] = hw[i,j] + delta_hidden[i,j] * learning_speed
			
	w = []
	w.append(hw)
	w.append(ow)

	return w, error
	
def agregate(weights, net_parameters):

	wa = net_parameters['wa']
	wb = net_parameters['wb']
	
	hweights = weights[0]
	oweights = weights[1]
	hl_size = np.shape(hweights)
	
	l = [0.0]*hl_size[1]
	for e in range(len(l)):
		l[e] = np.random.uniform(wa,wb)
	hweights = np.append(hweights, [l], axis = 0)
	oweights = np.append(oweights, [[1.0]], axis = 1)
	
	w = []
	w.append(hweights)
	w.append(oweights)
	
	return w
	
def make_zero(weights, learn_parameters):

	reduce_coeff = learn_parameters['reduction coefficient']

	hweights = weights[0]
	oweights = weights[1]
	
	size_hw = np.shape(hweights)
	size_ow = np.shape(oweights)

	for i in range(size_ow[0]):
		for j in range(size_ow[1]):
			if abs(oweights[i,j]) < abs((reduce_coeff*oweights.sum()))/(size_ow[0]*size_ow[1]):
				oweights[i,j] = 0.0
						
	for i in range(size_hw[0]):
		for j in range(size_hw[1]):
			if abs(hweights[i,j]) < abs((reduce_coeff*hweights.sum()))/(size_hw[0]*size_hw[1]):
				hweights[i,j] = 0.0		
	
	w = []
	w.append(hweights)
	w.append(oweights)
	
	return w
	
def reduce(weights):
	hweights = weights[0]
	oweights = weights[1]
	
	hweights = hweights[oweights.astype(bool).any(axis=0)]
	
	oweights = oweights[:, hweights.astype(bool).any(axis=1)]
	
	w=[]
	
	w.append(hweights)
	w.append(oweights)
	
	return w
	
def plot_data(data, t, xl, yl, c):
	plt.plot(data, c)
	plt.title(t)
	plt.xlabel(xl)
	plt.ylabel(yl)
	plt.show()
	
	
def learn(vectors, targets, weights, learn_parameters, net_parameters):
	
	test_vectors, test_targets = loadDataSet('spect_test.txt', delimiter = ',')
	
	max_iterations = learn_parameters['max iterations']
	threshold = learn_parameters['threshold']
	agr_coeff = learn_parameters['aggregation coefficient']
	
	err_list = []
	neurons_list = []
	
	time = 0
	eold=len(test_vectors)
	errold = len(vectors)
	k=0               # iterations index
	av_ms_error = 1.0 # average mean square error
	print threshold
	while k < max_iterations and av_ms_error > threshold:
		sum_error = 0.0
		for vector, target in zip(vectors, targets):
			net_output = run(vector, weights)
			weights, ms_error = backpropagate(vector, target, net_output, weights, learn_parameters)
			
			sum_error += ms_error # накопление ошибки
			
		# fix aggregation/reduction cool-down
		if abs((errold - sum_error)/errold) < agr_coeff and k > 2:
			weights = agregate(weights, net_parameters)	
			
		
				
		weights = make_zero(weights, learn_parameters)
			
		if k > 2 and np.shape(weights[0])[0] > 1:
				weights = reduce(weights)
			
		errold = sum_error
			
		
		
		# track errors and neurons for plots
		err_list.append(ms_error)
		neurons_list.append(np.shape(weights[0])[0])
		
		# calculate average mean square error
		av_ms_error = sum_error/len(vectors)
		
		print "cicle %d: neurons: %d ERROR: %.3f AVERAGE ERROR: %.3f" % (k, np.shape(weights[0])[0], sum_error, av_ms_error)
		k+=1
		
	plot_data(err_list, "error dynamics", "iterations", "error", "r")
	plot_data(neurons_list, "structure dynamics", "iterations", "neurons", "go")
	
	return weights
	
	
	
	
def test(vectors, targets, weights):
	errors_count = 0
	missclassified = []
	false_min = 0
	false_plus = 0
	
	for vector, target in zip(vectors, targets):
		d = run(vector, weights)[0]
		dr = d
		if d >= 0.5:
			d = 1
		else: d = 0
		if d != target:
			errors_count += 1
			if (target - d == 1):
				false_min += 1
			elif (target - d == -1):
				false_plus += 1
				
		missclassified.append(vector)
		missclassified.append(target)
		missclassified.append(d)
			
	net_performance = 100 - (errors_count * 100 / len(vectors))
	print "\nNumber of neurons: %d" % np.shape(weights[0])[0]
	print "Number of vectors: %d\tNumber of errors: %d false plus: %d false minus: %d" % (len(vectors), errors_count, false_plus, false_min)
	print "Net Performance is %f percent." % net_performance
	print "\nMissclassified vectors:"
	
	return net_performance, errors_count
		
def main():
		
	menu = {}
	menu['1'] = "Start"
	menu['2'] = "Resume"
	menu['3'] = "Test"
	menu['4'] = "Help"
	menu['5'] = "Exit application"
	
	while True:
		options=menu.keys()
		options.sort()
		print "--" * 10
		for entry in options:
			print entry, menu[entry]
			
		selection=raw_input("\nPlease, select: ")
		if selection == '1':
			file_name, delimiter = int.load_interface()  			 # run file load interface
			vectors, targets = loadDataSet(file_name, delimiter)
			net_parameters = int.net_init_interface()
			weights = init_mlp(net_parameters)
			learn_parameters = int.learn_interface()
			weights = learn(vectors, targets, weights, learn_parameters, net_parameters)
			choice = int.learn_ended_interface()
		elif selection == '2':
			try:
				print "current file: ", file_name
				weights = learn(vectors, targets, weights, learn_parameters, net_parameters)
			except:
				print "\nError"
		elif selection == '3':
			try:
				file_name, delimiter = int.load_interface()
				print "current file: ", file_name
				vectors, targets = loadDataSet(file_name, delimiter)
				test(vectors, targets, weights)
			except:
				print "\nError"
		elif selection == "4":
			readme.show_info()
		elif selection == '5':
			break
		else:
			print "No such option."
	
main()
		