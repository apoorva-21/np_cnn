import numpy as np

X = np.arange(1,26)

X = np.reshape(X, (5,5,1))
y = np.arange(1,9)
y = np.reshape(y, (y.shape[0],1))

W = np.ones((3,3,1))
# b = np.random.rand()
b = 0

#can use npad for padding

def conv_fwd(X, W, b, stride = 1, pad = 0):
	
	cache = (X,W)

	in_height = X.shape[0]
	in_width = X.shape[1]
	in_depth = X.shape[2]

	conv_height = W.shape[0]
	conv_width = W.shape[1]
	conv_depth = in_depth
	
	n_filt = 1

	#flatten the weight matrix to a vector
	W_flat = np.reshape(W, (conv_width * conv_height * conv_depth, n_filt))

	out_width = 1 + np.int(np.floor((in_width - conv_width + 2 * pad) / stride))
	out_height = 1 + np.int(np.floor((in_height - conv_height + 2 * pad) / stride))
	pos_h = 0
	pos_w = 0

	#flatten each under_the_window section into a flat vector
	#create the matrix with each row as one of the vectors
	flat_segments = []
	while pos_h + conv_height <= in_height:
		pos_w = 0
		while pos_w + conv_width <= in_width:
			in_seg = X[pos_h:pos_h + conv_height, pos_w:pos_w + conv_width,:]
			in_seg_flat = np.reshape(in_seg, (conv_height * conv_width * conv_depth, ))
			flat_segments.append(in_seg_flat)
			pos_w += stride
		pos_h += stride
	flat_segments = np.array(flat_segments)
	#use dot product logic for the forward prop on flattened segments
	out = np.dot(flat_segments, W_flat) + b

	#reshape the output to the expected output shape
	out = np.reshape(out, (out_height,out_width))
	return out, cache

def conv_back(d_out, cache):
	X, W = cache
	#for d_W use d_out as the filter and convolve over the input X from the cache

	# X = np.ones((3,3,1))
	# d_out = np.ones((2,2,1))
	# W = np.ones((2,2,1))

	d_W, _ = conv_fwd(X, d_out, b = 0)

	#for d_X we need a full convolution : need to pad the filter with zeros of filter_width - 1 and filter_height - 1

	pad_width = d_out.shape[1] - 1
	pad_height = d_out.shape[0] - 1

	lr_pad = np.zeros((W.shape[0], pad_width, W.shape[2]))
	print (lr_pad.shape, W.shape)
	W_padded_lr = np.hstack([lr_pad, W, lr_pad])

	ud_pad = np.zeros((pad_height,W_padded_lr.shape[1],W.shape[2]))
	W_padded_ud = np.vstack([ud_pad,W_padded_lr,ud_pad])

	#AND flip the d_out vertically and horizontally
	d_out_flip = np.flip(d_out,(0,1))

	d_X, _ = conv_fwd(W_padded_ud, d_out_flip, 0)

	return d_X, d_W

def relu(x):
	z = x.copy()
	z[x<0] = 0
	return z

def relu_back(x):
	z = np.ones_like(x)
	z[x<0] = 0
	z[x>0] = 1
	return z
#def softmax
#def softmax_back


#take train images

#define layers etc

#write training loop