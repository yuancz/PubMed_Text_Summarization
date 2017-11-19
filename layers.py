import tensorflow as tf
import numpy as np
import functions

def spatial_concat(inputs):
    inputs_list = tf.unstack(inputs, axis=1)
    concat_states = []
    for i in range(len(inputs_list)):
        if i == 0:
            concat_state = tf.concat([inputs_list[i], inputs_list[i], inputs_list[i+1]], axis=-1)
        elif i == len(inputs_list)-1:
            concat_state = tf.concat([inputs_list[i-1], inputs_list[i], inputs_list[i]], axis=-1)
        else:
            concat_state = tf.concat([inputs_list[i-1], inputs_list[i], inputs_list[i+1]], axis=-1)
        concat_states.append(concat_state)
    concat_states = tf.stack(concat_states, axis=1)
    return concat_states

def attention(inputs, W, v, input_lengths, hidden_size):
    inputs_W = tf.tanh(tf.reshape(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W),
                                  [-1, tf.shape(inputs)[1], hidden_size]))
    attn_weights = tf.reduce_sum(tf.multiply(v, inputs_W), 2)
    weights_mask = tf.sequence_mask(input_lengths, maxlen=tf.to_int32(tf.shape(attn_weights)[1]), dtype=tf.float32)
    attn_weights = attn_weights * weights_mask + ((1.0 - weights_mask) * tf.float32.min)
    attn_weights = tf.nn.softmax(attn_weights)
    inputs_attn = tf.multiply(tf.expand_dims(attn_weights, 2), inputs)
    attn_state = tf.reshape(tf.reduce_sum(inputs_attn, 1), [-1, hidden_size])
    return attn_state

def embeddings_layer(vocab_size, embedding_dim, trainable=True, pretrained_embeddings=None, name='embeddings'):
    if pretrained_embeddings is not None:
        embeddings = tf.get_variable(shape=pretrained_embeddings.shape,
                                     initializer=tf.constant_initializer(pretrained_embeddings),
                                     trainable=trainable,
                                     name=name)
    else:
        embeddings = tf.Variable(tf.truncated_normal([vocab_size, embedding_dim], stddev=0.1),
                                 name=name)
    return embeddings

def conv_1d(inputs, filter_size, in_channels, out_channels, stride=1, padding='SAME', name='conv1d'):
    filter_shape = [filter_size, in_channels, out_channels]
    
    W = tf.get_variable(name='W_{}'.format(name), initializer=tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.get_variable(name='b_{}'.format(name), initializer=tf.constant(0.1, shape=[out_channels]))
    
    conv = tf.nn.conv1d(inputs, W, stride=stride, padding=padding)
    return tf.nn.bias_add(conv, b)

def attention_layer(hidden_states, attn_size, attn_mask=None):
    U_i = tf.layers.dense(hidden_states, attn_size, use_bias=True, activation=tf.tanh,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          bias_initializer=tf.contrib.layers.xavier_initializer(),
                          name='U_i')
    U = tf.layers.dense(U_i, 1, use_bias=False,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name='U')
    U = tf.squeeze(U, axis=2)
    alpha_i = tf.nn.softmax(U)
    if attn_mask is not None:
        alpha_i = functions.masked_softmax(alpha_i, attn_mask)
    s_i = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha_i, -1), hidden_states), axis=1)
    return s_i

def match_GRU(c_inputs, q_inputs, hidden_size, dropout_keep_prob=1.0):
    W_c = tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size], stddev=0.1))
    W_q = tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size], stddev=0.1))
    W_o = tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size], stddev=0.1))
    w_alpha = tf.Variable(tf.truncated_normal(shape=[hidden_size, 1], stddev=0.1))
    b_c = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
    b_alpha = tf.Variable(tf.constant(0.1, shape=[1]))
    
    batch_size = tf.shape(c_inputs)[0]
    max_query_len = tf.shape(q_inputs)[1]
    
    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=dropout_keep_prob)
    init_state = cell.zero_state(batch_size, tf.float32)
    h_o = init_state
    
    H_c = tf.unstack(c_inputs, axis=1)
    H_q = tf.reshape(q_inputs, [-1, hidden_size])
    WH_q = tf.matmul(H_q, W_q)
    
    outputs = []
    for i in range(len(H_c)):
        h_c = H_c[i]
        G_tile = tf.tile((tf.matmul(h_c, W_c) + tf.matmul(h_o, W_o) + b_c), [max_query_len, 1])
        G = tf.tanh(WH_q + G_tile)
        
        a_tile = tf.tile(b_alpha, [max_query_len])
        alpha = tf.nn.softmax(tf.reshape(tf.matmul(G, w_alpha), [-1, max_query_len]) + a_tile)
        alpha = tf.expand_dims(alpha, 1)
        
        H_q = tf.reshape(H_q, [-1, max_query_len, hidden_size])
        z = tf.concat([h_c, tf.squeeze(tf.matmul(alpha, H_q), 1)], 1)
        
        output, h_o = cell(z, h_o)
        tf.get_variable_scope().reuse_variables()
        outputs.append(output)
    final_state = h_o
    outputs = tf.stack(outputs, axis=1)
    return outputs, final_state