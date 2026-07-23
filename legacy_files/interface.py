import scipy as sp

def load_interface():
	while True:
		try:
			fn = raw_input("filename>") #input filename
			d = raw_input("delimiter>") #input delimiter
	
			print "\nSuccess."
		except IOError:
			print "No such file. Try again..."
			continue
	
		return fn, d

def net_init_interface():
	while True:
		try:
			print "\nInput network parameters.\n"
			dict = {}
			dict['input layer size'] = abs(int(raw_input("input layer size>")))
			dict['hidden layer size'] = abs(int(raw_input("hidden layer size>")))
			dict['output layer size'] = abs(int(raw_input("output layer size>")))
			print "\nweights range (a,b):"
			dict['wa'] = int(raw_input("a>"))
			dict['wb'] = int(raw_input("b>"))
			
		except ValueError:
			print "\nWrong value. Try again..."
			continue
		
		return dict
		
def learn_interface():
	while True:
		try:
			print "\nInput learning parameters.\n"
			dict={}
			dict['learning speed'] = abs(float(raw_input("learning speed>")))
			dict['threshold'] = abs(float(raw_input("threshold>")))
			dict['max iterations'] = abs(int(raw_input("max iterations>")))
			dict['aggregation coefficient'] = abs(float(raw_input("aggregation coefficient>")))
			dict['reduction coefficient'] = abs(float(raw_input("reduction coefficient>")))
		
		except ValueError:
			print "Wrong value. Try again..."
			continue
			
		return dict
	