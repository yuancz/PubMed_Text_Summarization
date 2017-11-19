import tensorflow as tf

def run_layer(layer, states, in_channel, is_training, dropout_keep_prob):
    if type(layer) == Residual_Block:
        states = layer(states, in_channel, is_training, dropout_keep_prob)
    elif type(layer) == Conv:
        states = layer(states, in_channel)
    elif type(layer) == Pool:
        states = layer(states)
    elif type(layer) == Batchnorm:
        states = layer(states, is_training)
    elif type(layer) == Dense:
        states = layer(states)
    elif type(layer) == Maxpool_1d:
        states = layer(states)
    elif type(layer) == Concat:
        states = layer(states, in_channel, is_training, dropout_keep_prob)
    elif type(layer) == Dropout:
        states = layer(states, dropout_keep_prob)
    else:
        print('Layer not found')
    return states

class Conv:
    def __init__(self, filter_size, filter_depth, stride=1, padding='SAME', activation=tf.nn.relu, dim=1):
        self.filter_size = filter_size
        self.filter_depth = filter_depth
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.dim = dim
        
    def __call__(self, inputs, in_channel):
        if self.dim == 1:
            filter_shape = [self.filter_size, in_channel, self.filter_depth]
            W = tf.get_variable(name='conv1d_W', initializer=tf.contrib.layers.xavier_initializer(), 
                                shape=filter_shape)
            b = tf.get_variable(name='conv1d_b', initializer=tf.zeros_initializer(), shape=[self.filter_depth])
            
            outputs = tf.nn.conv1d(inputs, W, stride=self.stride, padding=self.padding)
        elif self.dim == 2:
            filter_shape = [self.filter_size, self.filter_size, in_channel, self.filter_depth]
            W = tf.get_variable(name='conv2d_W', initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                shape=filter_shape)
            b = tf.get_variable(name='conv2d_b', initializer=tf.zeros_initializer(), shape=[self.filter_depth])
            
            outputs = tf.nn.conv2d(inputs, W, [1, self.stride, self.stride, 1], self.padding)
        return self.activation(tf.nn.bias_add(outputs, b))

class Deconv:
    def __init__(self, filter_size, filter_depth, stride=1, padding='SAME', dim=1):
        self.filter_size = filter_size
        self.filter_depth = filter_depth
        self.stride = stride
        self.padding = padding
        self.dim = dim
        
    def __call__(self, inputs):
        if self.dim == 1:
            filter_shape = [self.filter_size, int(inputs.shape[-1])]
        elif self.dim == 2:
            filter_shape = [self.filter_size, self.filter_size]
        return conv2d_transpose(inputs, self.filter_depth, filter_shape, stride=self.stride, padding=self.padding,
                                activation=tf.nn.relu, use_bias=True, 
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    
class Pool:
    def __init__(self, filter_size, stride=2, padding='SAME', pool_type='MAX', dim=1):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = 'SAME'
        self.pool_type = 'MAX'
        self.dim = dim
        
    def __call__(self, inputs):
        if self.dim == 1:
            states = tf.expand_dims(inputs, 1)
            states = tf.nn.max_pool(states, [1, 1, self.filter_size, 1], [1, 1, self.stride, 1], self.padding)
            outputs = tf.squeeze(states, 1)
        elif self.dim == 2:
            outputs = tf.nn.max_pool(inputs, [1, self.filter_size, self.filter_size, 1], [1, self.stride, self.stride, 1],
                                     self.padding)
        return outputs

class Maxpool_1d:
    def __init__(self, dim=1):
        self.dim = dim
        
    def __call__(self, inputs):
        filter_w = int(inputs.shape[1])
        filter_h = int(inputs.shape[2])
        if self.dim == 1:
            states = tf.expand_dims(inputs, 1)
            states = tf.nn.max_pool(states, [1, 1, filter_w, 1], [1, 1, 1, 1], 'VALID')
            outputs = tf.squeeze(tf.squeeze(states, 1), 1)
        elif self.dim == 2:
            outputs = tf.nn.max_pool(inputs, [1, filter_h, filter_w, 1], [1, 1, 1, 1], 'VALID')
        return outputs
    
class Batchnorm:
    def __init__(self, center=True, scale=True):
        self.center = center
        self.scale = scale
        
    def __call__(self, inputs, is_training):
        states = tf.contrib.layers.batch_norm(inputs, center=self.center, scale=self.scale, is_training=is_training)
        return states
        
class Dropout:
    def __init__(self):
        pass
        
    def __call__(self, inputs, dropout_keep_prob):
        return tf.nn.dropout(inputs, dropout_keep_prob)
    
class Residual_Block:
    def __init__(self, block):
        self.block = block
        
    def __call__(self, inputs, in_channel, is_training):
        states = inputs
        for i, layer in enumerate(self.block):
            with tf.variable_scope('res_block_{}'.format(i)):
                in_channel = int(states.shape[-1])
                states = run_layer(layer, states, in_channel, is_training, dropout_keep_prob)
        return inputs + states

class Concat:
    def __init__(self, blocks):
        self.blocks = blocks
        
    def __call__(self, inputs, in_channel, is_training, dropout_keep_prob):
        collect_states = []
        for i, block in enumerate(self.blocks):
            with tf.variable_scope('block_{}'.format(i)):
                if type(block) == list:
                    states = inputs
                    for j, layer in enumerate(block):
                        with tf.variable_scope('layer_{}'.format(j)):
                            in_channel = int(states.shape[-1])
                            states = run_layer(layer, states, in_channel, is_training, dropout_keep_prob)
                else:
                    in_channel = int(inputs.shape[-1])
                    states = run_layer(block, inputs, in_channel, is_training, dropout_keep_prob)
            collect_states.append(states)
        return tf.concat(collect_states, -1)
    
class Dense:
    def __init__(self, output_size, activation=tf.nn.relu, use_bias=True):
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        
    def __call__(self, inputs):
        if len(inputs.shape) == 2:
            in_channels = 1
            in_width = int(inputs.shape[1])
        elif len(inputs.shape) == 3:
            in_channels = int(inputs.shape[2])
            in_width = int(inputs.shape[1])
        W = tf.get_variable(name='dense_W', initializer=tf.contrib.layers.xavier_initializer(), 
                            shape=[in_width*in_channels, self.output_size])
        if self.use_bias == True:
            b = tf.get_variable(name='dense_b', initializer=tf.zeros_initializer(), shape=[self.output_size])
            return self.activation(tf.matmul(tf.reshape(inputs, [-1, in_width*in_channels]), W) + b)
        else:
            return self.activation(tf.matmul(tf.reshape(inputs, [-1, in_width*in_channels]), W))