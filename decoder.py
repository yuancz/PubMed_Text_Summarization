import tensorflow as tf
import random 
import layers

def RNN_decoder(dec_inputs, cell, go_var, output_W, output_b, embeddings, init_state, hidden_size, teacher_forcing=False,
                teacher_forcing_mask=None, sample_decoding=False):
    if teacher_forcing_mask is None:
        teacher_forcing_mask = tf.ones([tf.shape(dec_inputs)[1]])
    batch_size = tf.shape(dec_inputs)[0]
        
    dec_inputs = tf.unstack(dec_inputs, axis=1)
    h_states = []
    output_logits = []
    generated_words = []
    for i in range(len(dec_inputs)): 
        if i > 0:
            tf.get_variable_scope().reuse_variables()
            
        if i == 0:
            prev_word = tf.tile(tf.expand_dims(go_var, 0), [batch_size, 1])
            h, state = cell(prev_word, init_state)
        else:
            one_or_zero = tf.equal(teacher_forcing_mask[i], 1)
            teacher_forcing_step = tf.cond(one_or_zero, lambda: teacher_forcing, lambda: False)
            generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                                     lambda: tf.to_int32(tf.argmax(logits, axis=1)))
            prev_word = tf.cond(teacher_forcing_step, lambda: dec_inputs[i-1], 
                                lambda: tf.nn.embedding_lookup(embeddings, generated_word))

            h, state = cell(prev_word, state)
            generated_words.append(generated_word)
        h_states.append(h)  
        logits = tf.matmul(h, output_W) + output_b
        output_logits.append(logits)
    output_logits = tf.stack(output_logits, axis=1)
    h_states = tf.stack(h_states, axis=1)
    
    generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                             lambda: tf.to_int32(tf.argmax(logits, axis=1)))
    generated_words.append(generated_word)
    generated_words = tf.stack(generated_words, axis=1)
    return output_logits, h_states, generated_words 

def RNN_basic_attn_decoder(dec_inputs, cell, go_var, output_W, output_b, embeddings, init_state, attn_state, hidden_size, 
                           teacher_forcing=False, teacher_forcing_mask=None, sample_decoding=False):
    if teacher_forcing_mask is None:
        teacher_forcing_mask = tf.ones([tf.shape(dec_inputs)[1]])
    batch_size = tf.shape(dec_inputs)[0]

    dec_inputs = tf.unstack(dec_inputs, axis=1)
    h_states = []
    output_logits = []
    generated_words = []
    for i in range(len(dec_inputs)): 
        if i > 0:
            tf.get_variable_scope().reuse_variables()
            
        if i == 0:
            prev_word = tf.tile(tf.expand_dims(go_var, 0), [batch_size, 1])
            h, state = cell(prev_word, init_state)
        else:
            one_or_zero = tf.equal(teacher_forcing_mask[i], 1)
            teacher_forcing_step = tf.cond(one_or_zero, lambda: teacher_forcing, lambda: False)
            generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                                     lambda: tf.to_int32(tf.argmax(logits, axis=1)))
            prev_word = tf.cond(teacher_forcing_step, lambda: dec_inputs[i-1], 
                                lambda: tf.nn.embedding_lookup(embeddings, generated_word))
            
            h, state = cell(prev_word, state)
            generated_words.append(generated_word)
        h_states.append(h)  
        logits = tf.matmul(tf.concat([h, attn_state], axis=1), output_W) + output_b
        output_logits.append(logits)
    output_logits = tf.stack(output_logits, axis=1)
    h_states = tf.stack(h_states, axis=1)
    
    generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                             lambda: tf.to_int32(tf.argmax(logits, axis=1)))
    generated_words.append(generated_word)
    generated_words = tf.stack(generated_words, axis=1)
    return output_logits, h_states, generated_words 

def RNN_attn_decoder(dec_inputs, enc_outputs, cell, go_var, output_W, output_b, embeddings, init_state, hidden_size,
                     enc_lens=None, teacher_forcing=False, teacher_forcing_mask=None, sample_decoding=False):
    # attention weights
    attn_v = tf.Variable(tf.truncated_normal([hidden_size]), name='v_attn')
    attn_W1 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size]), name='W1_attn')
    attn_W2 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size]), name='W2_attn')
    
    # weighted encoder outputs
    enc_outputs_w = tf.reshape(tf.matmul(tf.reshape(enc_outputs, [-1, hidden_size]), attn_W1), 
                               [-1, tf.shape(enc_outputs)[1], hidden_size])
    
    if teacher_forcing_mask is None:
        teacher_forcing_mask = tf.ones([tf.shape(dec_inputs)[1]])
    batch_size = tf.shape(dec_inputs)[0]

    dec_inputs = tf.unstack(dec_inputs, axis=1)
    h_states = []
    output_logits = []
    generated_words = []
    for i in range(len(dec_inputs)): 
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        if i == 0:
            prev_word = tf.tile(tf.expand_dims(go_var, 0), [batch_size, 1])
            h, state = cell(prev_word, init_state)
        else:
            one_or_zero = tf.equal(teacher_forcing_mask[i], 1)
            teacher_forcing_step = tf.cond(one_or_zero, lambda: teacher_forcing, lambda: False)
            generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                                     lambda: tf.to_int32(tf.argmax(logits, axis=1)))
            prev_word = tf.cond(teacher_forcing_step, lambda: dec_inputs[i-1], 
                                lambda: tf.nn.embedding_lookup(embeddings, generated_word))
            
            h, state = cell(prev_word, state)
            generated_words.append(generated_word)
        with tf.variable_scope('attention'):
            # weight the hidden state and combine with the weighted encoder outputs
            attn_query = tf.expand_dims(tf.matmul(h, attn_W2), 1)
            attn_tanh_sum = tf.tanh(tf.add(enc_outputs_w, attn_query))

            # calculate weighted sum to get a softmax distribution over the encoder outputs
            attention_weights = tf.reduce_sum(tf.multiply(attn_v, attn_tanh_sum), 2)
            if enc_lens is not None:
                num_weights = tf.shape(attention_weights)[1]
                weights_mask = tf.sequence_mask(lengths=enc_lens, maxlen=tf.to_int32(num_weights), dtype=tf.float32)
                attention_weights = attention_weights * weights_mask + ((1.0 - weights_mask) * tf.float32.min)
            attention_weights = tf.nn.softmax(attention_weights)                        

            # multiply softmax distribution with the encoder outputs 
            weighted_keys = tf.multiply(tf.expand_dims(attention_weights, 2), enc_outputs)
            
            # sum the weighted encoder outputs into a single hidden state vector
            context_h = tf.reshape(tf.reduce_sum(weighted_keys, 1), [-1, hidden_size])
            
        # project the attn hidden state into the vocab dimension to predict the next word
        logits = tf.matmul(tf.concat([h, context_h], axis=1), output_W) + output_b
        h_states.append(context_h)
        output_logits.append(logits)
    h_states = tf.stack(h_states, axis=1)
    output_logits = tf.stack(output_logits, axis=1)
    
    generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                             lambda: tf.to_int32(tf.argmax(logits, axis=1)))
    generated_words.append(generated_word)
    generated_words = tf.stack(generated_words, axis=1)
    return output_logits, h_states, generated_words  

def fusion_decoder(dec_inputs, lm_cell, dec_cell, go_var, output_W, output_b, embeddings, lm_init_state, dec_init_state, 
                   hidden_size, teacher_forcing=False):
    batch_size = tf.shape(dec_inputs)[0]
    lm_W = tf.Variable(tf.truncated_normal([hidden_size, hidden_size]), name='lm_W')
    lm_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='lm_b')
    g_W = tf.Variable(tf.truncated_normal([2*hidden_size, hidden_size]), name='g_W')
    g_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='g_b')
    #dec_W = tf.Variable(tf.truncated_normal([hidden_size, hidden_size]), name='dec_W')
    #dec_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='dec_b')
    fusion_W = tf.Variable(tf.truncated_normal([2*hidden_size, hidden_size]), name='fusion_W')
    fusion_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='fusion_b')
    
    dec_state = dec_init_state
    lm_state = lm_init_state
        
    dec_inputs = tf.unstack(dec_inputs, axis=1)
    h_states = []
    output_logits = []
    for i in range(len(dec_inputs)): 
        if i > 0:
            tf.get_variable_scope().reuse_variables()
            
        if i == 0:
            prev_word = tf.tile(tf.expand_dims(go_var, 0), [batch_size, 1])
            h, dec_state = dec_cell(prev_word, dec_state)
        else:
            prev_word = tf.cond(teacher_forcing, lambda: dec_inputs[i-1], 
                                lambda: tf.nn.embedding_lookup(embeddings, tf.argmax(tf.nn.softmax(logits), axis=1)))
            dec_h, dec_state = dec_cell(prev_word, dec_state)
            lm_h, lm_state = lm_cell(prev_word, lm_state)
            lm_h_W = tf.tanh(tf.matmul(lm_h, lm_W) + lm_b)
            g = tf.nn.sigmoid(tf.matmul(tf.concat([dec_h, lm_h_W], axis=1), g_W) + g_b)
            h = tf.tanh(tf.matmul(tf.concat([dec_h, tf.multiply(g, lm_h)], axis=1), fusion_W) + fusion_b)
        h_states.append(h)  
        logits = tf.matmul(h, output_W) + output_b
        output_logits.append(logits)
    output_logits = tf.stack(output_logits, axis=1)
    return output_logits, h_states

def fusion_attn_decoder(dec_inputs, lm_cell, dec_cell, go_var, output_W, output_b, embeddings, lm_init_state, dec_init_state, 
                        attn_h, hidden_size, teacher_forcing=False):
    batch_size = tf.shape(dec_inputs)[0]
    lm_W = tf.Variable(tf.truncated_normal([hidden_size, hidden_size]), name='lm_W')
    lm_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='lm_b')
    #dec_W = tf.Variable(tf.truncated_normal([hidden_size, hidden_size]), name='dec_W')
    #dec_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='dec_b')
    g_W = tf.Variable(tf.truncated_normal([2*hidden_size, hidden_size]), name='g_W')
    g_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='g_b')
    fusion_W = tf.Variable(tf.truncated_normal([2*hidden_size, hidden_size]), name='fusion_W')
    fusion_b = tf.Variable(tf.constant(0.0, shape=[hidden_size]), name='fusion_b')
    
    dec_state = dec_init_state
    lm_state = lm_init_state
        
    dec_inputs = tf.unstack(dec_inputs, axis=1)
    h_states = []
    output_logits = []
    for i in range(len(dec_inputs)): 
        if i > 0:
            tf.get_variable_scope().reuse_variables()
            
        if i == 0:
            prev_word = tf.tile(tf.expand_dims(go_var, 0), [batch_size, 1])
            h, dec_state = dec_cell(prev_word, dec_state)
        else:
            prev_word = tf.cond(teacher_forcing, lambda: dec_inputs[i-1], 
                                lambda: tf.nn.embedding_lookup(embeddings, tf.argmax(tf.nn.softmax(logits), axis=1)))
            dec_h, dec_state = dec_cell(prev_word, dec_state)
            lm_h, lm_state = lm_cell(prev_word, lm_state)
            lm_h_W = tf.nn.relu(tf.matmul(lm_h, lm_W) + lm_b)
            g = tf.nn.sigmoid(tf.matmul(tf.concat([dec_h, lm_h_W], axis=1), g_W) + g_b)
            h = tf.nn.relu(tf.matmul(tf.concat([dec_h, tf.multiply(g, lm_h)], axis=1), fusion_W) + fusion_b)
        h_states.append(h)  
        logits = tf.matmul(tf.concat([h, attn_h], axis=1), output_W) + output_b
        output_logits.append(logits)
    output_logits = tf.stack(output_logits, axis=1)
    return output_logits, h_states
    
class Attention_Cell:
    def __init__(enc_outputs, attn_size=None):
        self._input_size = tf.shape(enc_outputs)[-1]
        if attn_size is None:
            self._attn_size = self._input_size
        self._attn_v = tf.Variable(tf.truncated_normal([self._attn_size]), name='v_attn')
        self._attn_W1 = tf.Variable(tf.truncated_normal([self._input_size, self._attn_size]), name='W1_attn')
        self._attn_W2 = tf.Variable(tf.truncated_normal([self._input_size, self._attn_size]), name='W2_attn')
        
        self._enc_outputs = enc_outputs
        self._enc_outputs_w = tf.reshape(tf.matmul(tf.reshape(self._enc_outputs, [-1, self._input_size], self._attn_W1), 
                                                   [-1, tf.shape(self._enc_outputs)[1], self._attn_size]))
        
    def __call__(self, h, enc_lens):
        attn_query = tf.expand_dims(tf.matmul(h, self._attn_W2), 1)
        attn_tanh_sum = tf.tanh(tf.add(self._enc_outputs_w, attn_query))

        attention_weights = tf.reduce_sum(tf.multiply(self._attn_v, attn_tanh_sum), 2)
        num_weights = tf.shape(attention_weights)[1]
        weights_mask = tf.sequence_mask(lengths=enc_lens,
                                        maxlen=tf.to_int32(num_weights),
                                        dtype=tf.float32)
        attention_weights = attention_weights * weights_mask + ((1.0 - weights_mask) * tf.float32.min)
        attention_weights = tf.nn.softmax(attention_weights)                        

        weighted_keys = tf.multiply(tf.expand_dims(attention_weights, 2), self._enc_outputs)
        context_h = tf.reshape(tf.reduce_sum(weighted_keys, 1), [-1, self._input_size])
        return context_h, attention_weights