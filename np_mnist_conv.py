import os
import numpy as np
import pickle
import cv2

# np.seterr(all='print')

BATCH_SIZE = 200
LR = 0.03
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

#CONV FUNCTIONS ::
def conv_fwd(X, W, b, stride = 1, pad = 0, back = False):
	# print('FORWARD ::')
	# print('X = ',X.shape, W.shape)	
	cache = (X,W)
	in_height = X.shape[0]
	in_width = X.shape[1]
	in_depth = X.shape[2]

	conv_height = W.shape[0]
	conv_width = W.shape[1]
	
	conv_depth = W.shape[2]

	#flatten the weight matrix to a vector
	W_flat = np.reshape(W, (conv_width * conv_height * conv_depth))
	out_width = 1 + np.int(np.floor((in_width - conv_width + 2 * pad) / stride))
	out_height = 1 + np.int(np.floor((in_height - conv_height + 2 * pad) / stride))
	out_depth = 1
	pos_h = 0
	pos_w = 0

	#flatten each under_the_window section into a flat vector
	#create the matrix with each row as one of the vectors
	flat_segments = []
	while pos_h + conv_height <= in_height:
		pos_w = 0
		while pos_w + conv_width <= in_width:
			in_seg = 0
			in_seg_flat = 0
			if back:
				in_seg = X[pos_h:pos_h + conv_height, pos_w:pos_w + conv_width,:]
				in_seg_flat = np.reshape(in_seg, (conv_height * conv_width * X.shape[2]))
			else:
				in_seg = X[pos_h:pos_h + conv_height, pos_w:pos_w + conv_width,:,:]
				in_seg_flat = np.reshape(in_seg, (conv_height * conv_width * X.shape[2], X.shape[3]))
			flat_segments.append(in_seg_flat)
			pos_w += stride
		pos_h += stride
	flat_segments = np.array(flat_segments)

	flat_segments = flat_segments.T

	# print('FLAT SEGMENT: ',flat_segments.shape)

	# print(W_flat.T.shape)
	#use dot product logic for the forward prop on flattened segments
	out = np.dot(W_flat.T,flat_segments) + b
	#reshape the output to the expected output shape
	out = np.reshape(out, (out_height,out_width, X.shape[3]))
	# print(out.shape)
	# exit()
	# print('*'*50)
	return out, cache

def backprop_conv_fwd(X, W, b, stride = 1, pad = 0):

	# print('BACKWARD ::')
	cache = (X,W)
	# print('X = ',X.shape, W.shape)
	in_height = X.shape[0]
	in_width = X.shape[1]
	in_depth = X.shape[2]

	conv_height = W.shape[0]
	conv_width = W.shape[1]
	
	conv_depth = W.shape[2]

	#flatten the weight matrix to a vector
	W_flat = np.reshape(W, (conv_width * conv_height * conv_depth))
	out_width = 1 + np.int(np.floor((in_width - conv_width + 2 * pad) / stride))
	out_height = 1 + np.int(np.floor((in_height - conv_height + 2 * pad) / stride))
	out_depth = 1
	pos_h = 0
	pos_w = 0
		
	#flatten each under_the_window section into a flat vector
	#create the matrix with each row as one of the vectors
	flat_segments = []
	while pos_h + conv_height <= in_height:
		pos_w = 0
		while pos_w + conv_width <= in_width:
			in_seg = X[pos_h:pos_h + conv_height, pos_w:pos_w + conv_width,:,:]
			in_seg_flat = np.reshape(in_seg, (conv_height * conv_width * X.shape[3], X.shape[2]))
			flat_segments.append(in_seg_flat)
			pos_w += stride
		pos_h += stride
	flat_segments = np.array(flat_segments)

	flat_segments = flat_segments.T
	# print('FLAT SEGMENTS : ', flat_segments.shape)

	#use dot product logic for the forward prop on flattened segments
	# print(W_flat.T.shape)

	out = np.dot(W_flat.T,flat_segments) + b

	#reshape the output to the expected output shape
	out = np.reshape(out, (out_height,out_width, X.shape[2]))
	# print(out.shape)
	return out, cache


def conv_back(d_out, cache):
	X, W = cache
	#for d_W use d_out as the filter and convolve over the input X from the cache

	# X = np.ones((3,3,1))
	# d_out = np.ones((2,2,1))
	# W = np.ones((2,2,1))

	d_W, _ = backprop_conv_fwd(X, d_out, b = np.array([[0]]))

	#for d_X we need a full convolution : 
	#need to pad the filter with zeros of filter_width - 1 and filter_height - 1
	'''
	pad_width = d_out.shape[1] - 1
	pad_height = d_out.shape[0] - 1

	lr_pad = np.zeros((W.shape[0], pad_width, W.shape[2]))
	# print (lr_pad.shape, W.shape)
	W_padded_lr = np.hstack([lr_pad, W, lr_pad])

	ud_pad = np.zeros((pad_height,W_padded_lr.shape[1],W.shape[2]))
	W_padded_ud = np.vstack([ud_pad,W_padded_lr,ud_pad])

	#AND flip the d_out vertically and horizontally
	d_out_flip = np.flip(d_out,(0,1))

	d_X, _ = conv_fwd(W_padded_ud, d_out_flip, 0, back = True)'''

	return d_W

	# return d_W


def forward_prop(X, W, b):
	a = dict()
	z = dict()
	cache = dict()
	a['IN'] = X
	# z['HL1'] = np.dot(W['HL1'],X) + b['HL1']
	# a['HL1'] = relu_fwd(z['HL1'])
	# print(a['HL1'].shape)
	
	z['conv_1'], cache['conv_1'] = conv_fwd(a['IN'], W['conv_1'], b['conv_1'])
	a['conv_1'] = relu_fwd(z['conv_1'])
	a['conv_1_flat'] = np.reshape(a['conv_1'],(a['conv_1'].shape[0] * a['conv_1'].shape[1], X.shape[3]))

	#z['HL2'] = np.dot(W['HL2'],a['conv_1_flat']) + b['HL2']
	#a['HL2'] = relu_fwd(z['HL2'])
	
	# print(a['HL2'].shape)
	
	z['OUT'] = np.dot(W['OUT'],a['conv_1_flat']) + b['OUT']
	# print(z['OUT'])
	a['OUT'] = softmax_fwd(z['OUT'])
	
	return a, cache

def backward_prop(a, y_true, cache):
	dW = dict()
	db = dict()

	dz_OUT = cross_entropy_softmax_gradient(a['OUT'], y_true)
	# print(np.sum(dz_OUT))

	dW['OUT'] = np.dot(dz_OUT, a['conv_1_flat'].T) #10x361
	db['OUT'] = np.sum(dz_OUT, axis = 1) #10x1
	db['OUT'] = np.reshape(db['OUT'], (db['OUT'].shape[0],1)) #10x1

	da_conv_1_flat = np.dot(W['OUT'].T, dz_OUT) #361xb
	dz_conv_1_flat = da_conv_1_flat * relu_back(a['conv_1_flat'])
	#unflatten for conv backprop
	dz_conv_1 = np.reshape(dz_conv_1_flat,(out_height,out_height,BATCH_SIZE))

	dW['conv_1']= conv_back(dz_conv_1, cache['conv_1']) #128x256
	# print('CONV BACK DONE!',dW['conv_1'].shape)

	db['conv_1'] = np.sum(dz_conv_1) #1x1
	# db['HL2'] = np.reshape(db['HL2'], (db['HL2'].shape[0],1)) #10x1

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

#reshape inputs to 28x28 shape
X = np.reshape(X, (28, 28, 1, X.shape[1]))

X_test = normalize(X_test)
X_test = X_test.T

X_test = np.reshape(X_test, (28, 28, 1, X_test.shape[1]))

# cv2.imshow('frame',np.reshape(X[:,1],(28,28,1)))
# cv2.waitKey(0)

y = one_hotify(y)
y = y.T

y_test = one_hotify(y_test)
y_test = y_test.T

X_train = X[:,:,:,:55000]
y_train = y[:,:55000]
X_val = X[:,:,:,55000:]
y_val = y[:,55000:]

n_inputs = 28
n_HL1 = 256

conv_1_width = 10
conv_1_height = 10
n_in_channels = 1
pad = 0
stride = 1
out_width = 1 + np.int(np.floor((n_inputs - conv_1_width + 2 * pad) / stride))
out_height = 1 + np.int(np.floor((n_inputs - conv_1_height + 2 * pad) / stride))
out_depth = 1
n_HL2 = out_width * out_height
n_OUT = 10

W = dict()
b = dict()


W['conv_1'] = np.random.normal(0,1/n_inputs, (conv_1_height, conv_1_width, n_in_channels))
b['conv_1'] = 0#np.zeros((1,1))

# W['HL2'] = np.random.normal(0, 1 / n_inputs, (n_HL2, n_inputs)) #* 0.01
# b['HL2'] = np.zeros((n_HL2,1))

W['OUT'] = np.random.normal(0, 1 / n_HL2, (n_OUT, n_HL2)) #* 0.01
b['OUT'] = np.zeros((n_OUT,1))

# print(W)
for i in range(N_EPOCHS):
	avg_grad_per_epoch = 0
	for j in range(0, X_train.shape[1], BATCH_SIZE):
		X_batch = X_train[:,:,:,:BATCH_SIZE]
		y_batch = y_train[:,:BATCH_SIZE]

		#run forward prop on batch:
		a, cache = forward_prop(X_batch, W, b)

		#send the predictions to backprop:
		dW, db, dz_OUT= backward_prop(a, y_batch, cache)
		# avg_grad_per_epoch += dz_OUT
		#update via gradient descent:
		# if(j % 500 == 0):
			# print('BATCH NUMBER : {} PRED : {} TRUE : {}'.format(j / BATCH_SIZE, np.argmax(a['OUT'][:,0]),np.argmax(y_batch[:,0])))
		# print('*'*50)
		# print(W['HL1'])
		
		# W['HL1'] -= LR * dW['HL1']
		# b['HL1'] -= LR * db['HL1']

		W['conv_1'] -= LR * dW['conv_1']
		b['conv_1'] -= LR * db['conv_1']

		W['OUT'] -= LR * dW['OUT']
		b['OUT'] -= LR * db['OUT']


		# print('*'*50)
		
		# print(b['OUT'])
	# avg_grad_per_epoch = avg_grad_per_epoch / (X_train.shape[1]/ BATCH_SIZE)
	print('EPOCH : {}', i+1)# AVERAGE GRADIENT : {}'.format(i+1, avg_grad_per_epoch))
#validation:
print(X_test.shape)
a,_ = forward_prop(X_test, W, b)
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