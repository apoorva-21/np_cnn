import os
import numpy as np
import pickle
import cv2

# np.seterr(all='print')

BATCH_SIZE = 5500
LR = 0.9
N_EPOCHS = 20
k = -700
def normalize(X):
	return X / 255

def one_hotify(y):
	y_one_hot = np.eye(10)[y]
	return y_one_hot

def relu_fwd(x):
	z = np.zeros_like(x)
	z[x>0] = x[x>0]
	return z

def relu_back(x):
	z = np.zeros_like(x)
	z[x>0] = 1
	return z

def softmax_fwd(x):
	max_per_sample = np.max(x, axis = 0)
	max_per_sample = np.reshape(max_per_sample,(1,max_per_sample.shape[0]))
	x_sub_max = x - max_per_sample
	x_sub_max[x_sub_max < k] = k
	e_x = np.exp(x_sub_max) #10xb
	e_x_total = np.sum(e_x, axis = 0) #1xb
	sm_out = e_x / e_x_total
	return sm_out

def cross_entropy_softmax_gradient(a_out, y_true):
	#y_true should be one hot encoded
	return (a_out - y_true) / BATCH_SIZE

def forward_prop(X, W, b):
	a = dict()
	z = dict()

	a['IN'] = X
	# z['HL1'] = np.dot(W['HL1'],X) + b['HL1']
	# a['HL1'] = relu_fwd(z['HL1'])
	# print(a['HL1'].shape)

	z['HL2'] = np.dot(W['HL2'],a['IN']) + b['HL2']
	a['HL2'] = relu_fwd(z['HL2'])
	# print(a['HL2'].shape)
	
	z['OUT'] = np.dot(W['OUT'],a['HL2']) + b['OUT']
	# print(z['OUT'])
	a['OUT'] = softmax_fwd(z['OUT'])
	# print(a['OUT'].shape)

	return a

def backward_prop(a, y_true):
	dW = dict()
	db = dict()

	dz_OUT = cross_entropy_softmax_gradient(a['OUT'], y_true)
	# print(np.sum(dz_OUT))

	dW['OUT'] = np.dot(dz_OUT, a['HL2'].T) #10x128
	db['OUT'] = np.sum(dz_OUT, axis = 1) #10x1
	db['OUT'] = np.reshape(db['OUT'], (db['OUT'].shape[0],1)) #10x1

	da_HL2 = np.dot(W['OUT'].T, dz_OUT) #128xb
	dz_HL2 = da_HL2 * relu_back(a['HL2'])

	dW['HL2'] = np.dot(dz_HL2, a['IN'].T) #128x256
	db['HL2'] = np.sum(dz_HL2, axis = 1) #128x1
	db['HL2'] = np.reshape(db['HL2'], (db['HL2'].shape[0],1)) #10x1

	# da_HL1 = np.dot(W['HL2'].T, dz_HL2) #256xb
	# dz_HL1 = da_HL1 * relu_back(a['HL1'])

	# dW['HL1'] = np.dot(dz_HL1, a['IN'].T)
	# db['HL1'] = np.sum(dz_HL1, axis = 1)
	# db['HL1'] = np.reshape(db['HL1'], (db['HL1'].shape[0],1)) #10x1

	return dW, db, dz_OUT

DATA_PATH = './data/train_data.pickle'
LABELS_PATH = './data/train_labels.pickle'


TEST_DATA_PATH = './data/test_data.pickle'
TEST_LABELS_PATH = './data/test_labels.pickle'

X = []
y = []

X_test = []
y_test = []

with open(DATA_PATH, 'rb') as f:
	X = pickle.load(f)

with open(LABELS_PATH, 'rb') as f:
	y = pickle.load(f)


with open(TEST_DATA_PATH, 'rb') as f:
	X_test = pickle.load(f)

with open(TEST_LABELS_PATH, 'rb') as f:
	y_test = pickle.load(f)


X = normalize(X)
X = X.T

X_test = normalize(X_test)
X_test = X_test.T

# cv2.imshow('frame',np.reshape(X[:,1],(28,28,1)))
# cv2.waitKey(0)

y = one_hotify(y)
y = y.T

y_test = one_hotify(y_test)
y_test = y_test.T

X_train = X#[:,:55000]
y_train = y#[:,:55000]
# X_val = X[:,55000:]
# y_val = y[:,55000:]

n_inputs = 784
n_HL1 = 256
n_HL2 = 128
n_OUT = 10

W = dict()
b = dict()

# W['HL1'] = np.random.normal(0,1/n_inputs, (n_HL1, n_inputs)) #* 0.01
# b['HL1'] = np.zeros((n_HL1,1))

W['HL2'] = np.random.normal(0, 1 / n_inputs, (n_HL2, n_inputs)) #* 0.01
b['HL2'] = np.zeros((n_HL2,1))

W['OUT'] = np.random.normal(0, 1 / n_HL2, (n_OUT, n_HL2)) #* 0.01
b['OUT'] = np.zeros((n_OUT,1))

# print(W)
for i in range(N_EPOCHS):
	avg_grad_per_epoch = 0
	for j in range(0, X_train.shape[1], BATCH_SIZE):
		X_batch = X_train[:,: BATCH_SIZE]
		y_batch = y_train[:,:BATCH_SIZE]

		#run forward prop on batch:
		a = forward_prop(X_batch, W, b)

		#send the predictions to backprop:
		dW, db, dz_OUT= backward_prop(a, y_batch)
		# avg_grad_per_epoch += dz_OUT
		#update via gradient descent:
		# if(j % 500 == 0):
			# print('BATCH NUMBER : {} PRED : {} TRUE : {}'.format(j / BATCH_SIZE, np.argmax(a['OUT'][:,0]),np.argmax(y_batch[:,0])))
		# print('*'*50)
		# print(W['HL1'])
		
		# W['HL1'] -= LR * dW['HL1']
		# b['HL1'] -= LR * db['HL1']

		W['HL2'] -= LR * dW['HL2']
		b['HL2'] -= LR * db['HL2']

		W['OUT'] -= LR * dW['OUT']
		b['OUT'] -= LR * db['OUT']


		# print('*'*50)
		
		# print(b['OUT'])
	# avg_grad_per_epoch = avg_grad_per_epoch / (X_train.shape[1]/ BATCH_SIZE)
	print('EPOCH : {}', i+1)# AVERAGE GRADIENT : {}'.format(i+1, avg_grad_per_epoch))
#validation:

a = forward_prop(X_test, W, b)
y_pred = a['OUT']
print(y_pred.shape)
print(y_test.shape)
correct = 0.
for i in range(y_pred.shape[1]):
	pred = np.argmax(y_pred[:,i])
	true = np.argmax(y_test[:,i])
	if  pred == true:
		# print(i,np.argmax[:,i])
		correct+=1
	# else :
		# print(pred, true)
acc = correct * 1. / y_pred.shape[1]
print("Validation Accuracy = ", acc)