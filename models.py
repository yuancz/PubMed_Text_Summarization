import tensorflow as tf
import layers
import functions
import decoder
import encoder

class RNN_Classifier:
    def __init__(self, n_classes, vocab_size, max_len, embedding_dim=100, hidden_size=128, n_layers=1, 
                 bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.inputs = tf.placeholder(tf.int32, [None, self.max_len], name='inputs')
        self.input_lens = tf.placeholder(tf.int32, [None], name='input_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                      trainable=self.trainable_embeddings,
                                                      pretrained_embeddings=self._pretrained_embeddings,
                                                      name='embeddings')
            inputs_embd = tf.nn.embedding_lookup(self.embeddings, self.inputs)

        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
            
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.cell = gru_cell()
            outputs, final_output, final_state = encoder.RNN_encoder(inputs_embd, self.cell, self.hidden_size, 
                                                                     input_lens=self.input_lens)
            
        with tf.variable_scope('dense_output'):
            self.output_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_classes]), name='output_W')
            self.output_b = tf.Variable(tf.constant(0.0, shape=[self.n_classes]), name='output_b')
            
            self.logits = tf.matmul(final_output, self.output_W) + self.output_b

class Language_Model:
    def __init__(self, vocab_size, max_enc_len, embedding_dim=100, hidden_size=128, n_layers=1, 
                 pretrained_embeddings=None, trainable_embeddings=True):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.trainable_embeddings = trainable_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.seed_length = tf.placeholder(tf.int32, name='seed_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        self._build_model()
    
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                      trainable=self.trainable_embeddings,
                                                      pretrained_embeddings=self._pretrained_embeddings,
                                                      name='embeddings')
            inputs_embd = tf.nn.embedding_lookup(self.embeddings, self.enc_inputs)
        
        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('lm_encoder'):
            if self.n_layers > 1:
                self.cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.cell = gru_cell()
                
            self.output_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.vocab_size]), name='output_W')
            self.output_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='output_b')
            
            (self.h_states, 
             self.final_state, self.logits) = encoder.lm_encoder(inputs_embd, self.cell, self.output_W, self.output_b, 
                                                                 self.embeddings, self.seed_length, self.hidden_size,
                                                                 input_lens=self.enc_lens)

class Seq2Seq_CNN:
    def __init__(self, cnn_protobuf, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.cnn_protobuf = cnn_protobuf
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        self.teacher_forcing_mask = tf.placeholder(tf.int32, [self.max_dec_len], name='teacher_forcing_mask')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)

        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('CNN_encoder'):
            cnn_outputs = encoder.CNN_encoder(enc_inputs_embd, self.cnn_protobuf, is_training=self.is_training,
                                              dropout_keep_prob=self.dropout_keep_prob)
            print(cnn_outputs.get_shape())
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
                cnn_outputs = tuple(cnn_outputs for _ in range(self.n_layers))
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.basic_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                        self.dec_b, self.dec_embeddings, cnn_outputs,  
                                                        self.hidden_size, teacher_forcing=self.teacher_forcing,
                                                        teacher_forcing_mask=self.teacher_forcing_mask)
            
class Seq2Seq_Attn_CNN:
    def __init__(self, cnn_protobuf, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.cnn_protobuf = cnn_protobuf
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        self.teacher_forcing_mask = tf.placeholder(tf.int32, [self.max_dec_len], name='teacher_forcing_mask')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)

        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('CNN_encoder'):
            cnn_state = encoder.CNN_encoder(enc_inputs_embd, self.cnn_protobuf, is_training=self.is_training)
            attn_state = cnn_state
            print(cnn_state.get_shape())
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
                cnn_state = tuple(cnn_state for _ in range(self.n_layers))
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.basic_attn_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                             self.dec_b, self.dec_embeddings, cnn_state, attn_state, 
                                                             self.hidden_size, teacher_forcing=self.teacher_forcing,
                                                             teacher_forcing_mask=self.teacher_forcing_mask)

class Seq2Seq_Attn_CNN_RNN:
    def __init__(self, cnn_protobuf, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.cnn_protobuf = cnn_protobuf
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
                print(enc_inputs_embd.get_shape())

        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
                
        with tf.variable_scope('CNN_encoder'):
            cnn_outputs = encoder.CNN_encoder(enc_inputs_embd, self.cnn_protobuf)
            reduction_multiple = self.max_enc_len / tf.shape(cnn_outputs)[1]
            reduced_enc_lens = tf.to_int32(tf.div(tf.to_float(self.enc_lens), tf.to_float(reduction_multiple)))
            print(cnn_outputs.get_shape())
            
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = gru_cell()
                
            enc_outputs, final_state = encoder.basic_encoder(cnn_outputs, self.enc_cell, self.hidden_size, 
                                                             input_lens=reduced_enc_lens)
            print(enc_outputs.get_shape())
            final_state = final_state[-1] if self.n_layers > 1 else final_state
            
        with tf.variable_scope('attention'):
            self.attn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='attn_W')
            self.attn_v = tf.Variable(tf.truncated_normal([self.hidden_size]), name='attn_v')
            attn_state = layers.attention(enc_outputs, self.attn_W, self.attn_v, reduced_enc_lens, self.hidden_size)
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.basic_attn_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                             self.dec_b, self.dec_embeddings, final_state, attn_state, 
                                                             self.hidden_size, teacher_forcing=self.teacher_forcing)  

class Seq2Seq_Attn_HCNN_RNN:
    def __init__(self, cnn_protobuf, n_hiers, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.cnn_protobuf = cnn_protobuf
        self.n_hiers = n_hiers
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
                print(enc_inputs_embd.get_shape())
                
        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('HCNN_RNN'):
            states = enc_inputs_embd
            print(states.get_shape())
            lens = self.enc_lens
            for i in range(self.n_hiers):
                with tf.variable_scope('CNN_{}'.format(i)):
                    states = encoder.CNN_encoder(states, self.cnn_protobuf)
                    reduction_multiple = self.max_enc_len / tf.shape(states)[1]
                    lens = tf.to_int32(tf.div(tf.to_float(lens), tf.to_float(reduction_multiple)))
                    print(states.get_shape())

                with tf.variable_scope('RNN_{}'.format(i)):
                    if self.n_layers > 1:
                        self.enc_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
                    else:
                        self.enc_cell = gru_cell()

                    states, final_state = encoder.basic_encoder(states, self.enc_cell, self.hidden_size, 
                                                                input_lens=lens)
                    final_state = final_state[-1] if self.n_layers > 1 else final_state
            enc_outputs = states
            
        with tf.variable_scope('attention'):
            self.attn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='attn_W')
            self.attn_v = tf.Variable(tf.truncated_normal([self.hidden_size]), name='attn_v')
            attn_state = layers.attention(enc_outputs, self.attn_W, self.attn_v, lens, self.hidden_size)
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.basic_attn_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                             self.dec_b, self.dec_embeddings, final_state, attn_state, 
                                                             self.hidden_size, teacher_forcing=self.teacher_forcing)  

class Seq2Seq_Basic_Attn_Dual_CNN_RNN:
    def __init__(self, cnn_protobuf, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.cnn_protobuf = cnn_protobuf
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = gru_cell()
                
            enc_outputs, final_state = encoder.basic_encoder(enc_inputs_embd, self.enc_cell, self.hidden_size, 
                                                             input_lens=self.enc_lens)
            
            final_state = final_state[-1] if self.n_layers > 1 else final_state
            
        with tf.variable_scope('attention'):
            self.attn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='attn_W')
            self.attn_v = tf.Variable(tf.truncated_normal([self.hidden_size]), name='attn_v')
            attn_state = layers.attention(enc_outputs, self.attn_W, self.attn_v, self.enc_lens, self.hidden_size)
            
        with tf.variable_scope('CNN_encoder'):
            cnn_state = encoder.CNN_encoder(enc_inputs_embd, self.cnn_protobuf)
            print(cnn_state.get_shape())
            
        with tf.variable_scope('fusion'):
            self.cnn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='cnn_W')
            self.rnn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='rnn_W')
            self.fuse_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]), name='fuse_b')
            fuse_state = tf.tanh(tf.matmul(cnn_state, self.cnn_W) + tf.matmul(attn_state, self.rnn_W) + self.fuse_b)
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.basic_attn_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                             self.dec_b, self.dec_embeddings, final_state, fuse_state, 
                                                             self.hidden_size, teacher_forcing=self.teacher_forcing)

class Seq2Seq_Attn:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True, rnn_cell=tf.contrib.rnn.GRUCell):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self.rnn_cell = rnn_cell
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        self.teacher_forcing_mask = tf.placeholder(tf.int32, [self.max_dec_len], name='teacher_forcing_mask')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.sample_decoding = tf.placeholder(tf.bool, name='sample_decoding')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
            enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
            dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def cell():
            return tf.contrib.rnn.DropoutWrapper(self.rnn_cell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = cell()    
            enc_outputs, final_output, final_state = encoder.RNN_encoder(enc_inputs_embd, self.enc_cell, 
                                                                         self.hidden_size, input_lens=self.enc_lens)

        with tf.variable_scope('GRU_attn_decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = cell()
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            (self.logits, self.dec_states, 
             self.generated_words) = decoder.RNN_attn_decoder(dec_inputs_embd, enc_outputs, self.dec_cell, self.go_var, 
                                                              self.dec_W, self.dec_b, self.dec_embeddings, final_state, 
                                                              self.hidden_size, enc_lens=self.enc_lens, 
                                                              teacher_forcing=self.teacher_forcing, 
                                                              teacher_forcing_mask=self.teacher_forcing_mask, 
                                                              sample_decoding=self.sample_decoding)
            
class Seq2Seq_Basic_Attn:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True, weight_tying=False, rnn_cell=tf.contrib.rnn.GRUCell):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self.weight_tying = weight_tying
        self.rnn_cell = rnn_cell
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        self.teacher_forcing_mask = tf.placeholder(tf.int32, [self.max_dec_len], name='teacher_forcing_mask')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.sample_decoding = tf.placeholder(tf.bool, name='sample_decoding')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
            enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
            dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def cell():
            return tf.contrib.rnn.DropoutWrapper(self.rnn_cell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = cell()
            enc_outputs, final_output, final_state = encoder.RNN_encoder(enc_inputs_embd, self.enc_cell, 
                                                                         self.hidden_size, input_lens=self.enc_lens)
            
        with tf.variable_scope('attention'):
            self.attn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='attn_W')
            self.attn_v = tf.Variable(tf.truncated_normal([self.hidden_size]), name='attn_v')
            attn_state = layers.attention(enc_outputs, self.attn_W, self.attn_v, self.enc_lens, self.hidden_size)
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = cell()
            if self.weight_tying == True:
                proj_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.embedding_dim]), name='proj_W')
                self.dec_W = tf.tanh(tf.matmul(proj_W, tf.transpose(self.dec_embeddings, [1, 0])))
            else:
                self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')

            (self.logits, self.dec_states, 
             self.generated_words) = decoder.RNN_basic_attn_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                                    self.dec_b, self.dec_embeddings, final_state, attn_state, 
                                                                    self.hidden_size, teacher_forcing=self.teacher_forcing, 
                                                                    teacher_forcing_mask=self.teacher_forcing_mask, 
                                                                    sample_decoding=self.sample_decoding)

class Seq2Seq_Basic_C_Attn:
    def __init__(self, cnn_protobuf, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.cnn_protobuf = cnn_protobuf
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = gru_cell()
                
            enc_outputs, final_state = encoder.basic_encoder(enc_inputs_embd, self.enc_cell, self.hidden_size, 
                                                             input_lens=self.enc_lens)
            
            final_state = final_state[-1] if self.n_layers > 1 else final_state
            
        with tf.variable_scope('C_attention'):
            cnn_state = encoder.CNN_encoder(enc_outputs, self.cnn_protobuf, is_training=self.is_training,
                                            dropout_keep_prob=self.dropout_keep_prob)
            print(cnn_state.get_shape())
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.basic_attn_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                             self.dec_b, self.dec_embeddings, final_state, cnn_state, 
                                                             self.hidden_size, teacher_forcing=self.teacher_forcing)
            
class Seq2Seq_Basic_Attn_LM:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.seed_length = tf.placeholder(tf.int32, name='seed_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = gru_cell()
                
            enc_outputs, final_state = encoder.basic_encoder(enc_inputs_embd, self.enc_cell, self.hidden_size, 
                                                             input_lens=self.enc_lens)
            final_state = final_state[-1] if self.n_layers > 1 else final_state
            
        with tf.variable_scope('attention'):
            self.attn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='attn_W')
            self.attn_v = tf.Variable(tf.truncated_normal([self.hidden_size]), name='attn_v')
            attn_state = layers.attention(enc_outputs, self.attn_W, self.attn_v, self.enc_lens, self.hidden_size)
        
        with tf.variable_scope('lm_encoder'):
            if self.n_layers > 1:
                self.lm_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.lm_cell = gru_cell()
            self.lm_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.vocab_size]), name='dec_W')
            self.lm_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            
            _, lm_state, self.lm_logits = encoder.lm_encoder(enc_inputs_embd, self.lm_cell, self.lm_W, self.lm_b, 
                                                             self.enc_embeddings, self.seed_length, self.hidden_size,
                                                             input_lens=self.enc_lens)
        
        with tf.variable_scope('basic_attn_decoder'):
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.basic_attn_decoder(dec_inputs_embd, self.lm_cell, self.go_var, self.dec_W, 
                                                             self.dec_b, self.dec_embeddings, lm_state, attn_state, 
                                                             self.hidden_size, teacher_forcing=self.teacher_forcing)
            #self.logits, states = decoder.basic_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
            #                                            self.dec_b, self.dec_embeddings, attn_state, self.hidden_size,
            #                                            teacher_forcing=self.teacher_forcing)

class Seq2Seq_Attn_LM_Fusion:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.seed_length = tf.placeholder(tf.int32, name='seed_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = gru_cell()
                
            enc_outputs, final_state = encoder.basic_encoder(enc_inputs_embd, self.enc_cell, self.hidden_size, 
                                                             input_lens=self.enc_lens)
            final_state = final_state[-1] if self.n_layers > 1 else final_state
            
        with tf.variable_scope('attention'):
            self.attn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='attn_W')
            self.attn_v = tf.Variable(tf.truncated_normal([self.hidden_size]), name='attn_v')
            attn_state = layers.attention(enc_outputs, self.attn_W, self.attn_v, self.enc_lens, self.hidden_size)
            
        with tf.variable_scope('lm_encoder'):
            if self.n_layers > 1:
                self.lm_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.lm_cell = gru_cell()
            self.lm_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.vocab_size]), name='dec_W')
            self.lm_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            
            _, lm_state, self.lm_logits = encoder.lm_encoder(enc_inputs_embd, self.lm_cell, self.lm_W, self.lm_b, 
                                                             self.enc_embeddings, self.seed_length, self.hidden_size,
                                                             input_lens=self.enc_lens)
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.fusion_decoder(dec_inputs_embd, self.lm_cell, self.dec_cell, self.go_var, 
                                                         self.dec_W, self.dec_b, self.dec_embeddings, lm_state, final_state, 
                                                         attn_state, self.hidden_size, teacher_forcing=self.teacher_forcing)
            
class Seq2Seq_LM_Fusion:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.seed_length = tf.placeholder(tf.int32, name='seed_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = gru_cell()
                
            enc_outputs, final_state = encoder.basic_encoder(enc_inputs_embd, self.enc_cell, self.hidden_size, 
                                                             input_lens=self.enc_lens)
            final_state = final_state[-1] if self.n_layers > 1 else final_state

        with tf.variable_scope('lm_encoder'):
            if self.n_layers > 1:
                self.lm_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.lm_cell = gru_cell()
            self.lm_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.vocab_size]), name='dec_W')
            self.lm_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            
            _, lm_state, self.lm_logits = encoder.lm_encoder(enc_inputs_embd, self.lm_cell, self.lm_W, self.lm_b, 
                                                             self.enc_embeddings, self.seed_length, self.hidden_size,
                                                             input_lens=self.enc_lens)
          
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = gru_cell()
            self.dec_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.vocab_size]), name='W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')
            
            self.logits, states = decoder.fusion_decoder(dec_inputs_embd, self.lm_cell, self.dec_cell, self.go_var, 
                                                         self.dec_W, self.dec_b, self.dec_embeddings, lm_state, final_state, 
                                                         self.hidden_size, teacher_forcing=self.teacher_forcing)

class Seq2Seq_Single_Attn:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self.__pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        with tf.variable_scope('GRU_encoder'):
            enc_outputs, final_state = encoder.GRU_encoder(enc_inputs_embd, self.hidden_size, self.enc_lens,
                                                           self.dropout_keep_prob, n_layers=self.n_layers,
                                                           bidirectional=self.bidirectional)
            
        with tf.variable_scope('GRU_attn_decoder'):
            if self.bidirectional == True:
                dec_h_size = 2*self.hidden_size
            else:
                dec_h_size = self.hidden_size
            self.logits, states = decoder.GRU_single_attn_decoder(dec_inputs_embd, enc_outputs, final_state, dec_h_size, 
                                                                  self.vocab_size, self.dec_embeddings, self.embedding_dim, 
                                                                  self.max_dec_len, self.teacher_forcing, 
                                                                  self.dropout_keep_prob, self.enc_lens, 
                                                                  n_layers=self.n_layers)
            
class Seq2Seq_Pre_Attn:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, n_attns=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_attns = n_attns
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self.__pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        with tf.variable_scope('GRU_encoder'):
            enc_outputs, final_state = encoder.GRU_encoder(enc_inputs_embd, self.hidden_size, self.enc_lens,
                                                           self.dropout_keep_prob, n_layers=self.n_layers,
                                                           bidirectional=self.bidirectional)
            
        with tf.variable_scope('GRU_attn_decoder'):
            if self.bidirectional == True:
                dec_h_size = 2*self.hidden_size
            else:
                dec_h_size = self.hidden_size
            (self.logits, states, 
             self.attn_dists, self.final_attns) = decoder.GRU_pre_attn_decoder(dec_inputs_embd, enc_outputs, final_state, 
                                                                                dec_h_size, self.vocab_size, self.dec_embeddings, 
                                                                                self.embedding_dim, self.max_dec_len, 
                                                                                self.teacher_forcing, self.dropout_keep_prob, 
                                                                                self.enc_lens, n_layers=self.n_layers, 
                                                                                n_attns=self.n_attns)
            
class Seq2Seq:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        with tf.variable_scope('GRU_encoder'):
            enc_outputs, final_state = encoder.GRU_encoder(enc_inputs_embd, self.hidden_size, self.enc_lens,
                                                           self.dropout_keep_prob, n_layers=self.n_layers,
                                                           bidirectional=self.bidirectional)
            
        with tf.variable_scope('GRU_decoder'):
            if self.bidirectional == True:
                dec_h_size = 2*self.hidden_size
            else:
                dec_h_size = self.hidden_size
            self.logits, states = decoder.GRU_decoder(dec_inputs_embd, final_state, dec_h_size, self.vocab_size, 
                                                      self.dec_embeddings, self.embedding_dim, n_layers=self.n_layers, 
                                                      dropout_keep_prob=self.dropout_keep_prob, 
                                                      teacher_forcing=self.teacher_forcing)
            
class CGRU_Seq2Seq_Attn:
    def __init__(self, protobuf, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.protobuf = protobuf
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self.__pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            print(enc_inputs_embd.get_shape())
        with tf.variable_scope('CNN_encoder'):
            cnn_outputs = encoder.CNN_encoder(enc_inputs_embd, self.protobuf)
            
            reduction_multiple = self.max_enc_len / tf.shape(cnn_outputs)[1]
            reduced_enc_lens = tf.to_int32(tf.div(tf.to_float(self.enc_lens), tf.to_float(reduction_multiple)))
            print(cnn_outputs.get_shape())
        
        with tf.variable_scope('GRU_encoder'):
            enc_outputs, final_state = encoder.GRU_encoder(cnn_outputs, self.hidden_size, reduced_enc_lens,
                                                           self.dropout_keep_prob, n_layers=self.n_layers,
                                                           bidirectional=self.bidirectional)
            print(enc_outputs.get_shape())
            
        with tf.variable_scope('GRU_attn_decoder'):
            if self.bidirectional == True:
                dec_h_size = 2*self.hidden_size
            else:
                dec_h_size = self.hidden_size
            self.logits, states = decoder.GRU_attn_decoder(dec_inputs_embd, enc_outputs, final_state, dec_h_size, 
                                                           self.vocab_size, self.dec_embeddings, self.embedding_dim, 
                                                           self.max_dec_len, self.teacher_forcing, self.dropout_keep_prob, 
                                                           reduced_enc_lens, n_layers=self.n_layers)
            
class CGRU_Seq2Seq:
    def __init__(self, protobuf, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True):
        self.protobuf = protobuf
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self.__pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self.__pretrained_embeddings,
                                                              name='dec_embeddings')
                enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
                dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
            print(enc_inputs_embd.get_shape())
        with tf.variable_scope('CNN_encoder'):
            cnn_outputs = encoder.CNN_encoder(enc_inputs_embd, self.protobuf)
            
            reduction_multiple = self.max_enc_len / tf.shape(cnn_outputs)[1]
            reduced_enc_lens = tf.to_int32(tf.div(tf.to_float(self.enc_lens), tf.to_float(reduction_multiple)))
            print(cnn_outputs.get_shape())
        
        with tf.variable_scope('GRU_encoder'):
            enc_outputs, final_state = encoder.GRU_encoder(cnn_outputs, self.hidden_size, reduced_enc_lens,
                                                           self.dropout_keep_prob, n_layers=self.n_layers,
                                                           bidirectional=self.bidirectional)
            print(enc_outputs.get_shape())
            
        with tf.variable_scope('GRU_decoder'):
            if self.bidirectional == True:
                dec_h_size = 2*self.hidden_size
            else:
                dec_h_size = self.hidden_size
            self.logits, states = decoder.GRU_decoder(dec_inputs_embd, final_state, dec_h_size, self.vocab_size, 
                                                      self.dec_embeddings, self.embedding_dim, n_layers=self.n_layers, 
                                                      dropout_keep_prob=self.dropout_keep_prob, 
                                                      teacher_forcing=self.teacher_forcing)

class GRU_Net:
    def __init__(self, n_classes, vocab_size, max_sent_len, embedding_dim=100, hidden_size=128, n_layers=1,
                 bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.max_sent_len = max_sent_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.pretrained_embeddings = pretrained_embeddings
        self.trainable_embeddings = trainable_embeddings
        
        self.inputs = tf.placeholder(tf.int32, [None, self.max_sent_len], name='inputs')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.input_lens = tf.placeholder(tf.int32, [None], name='input_lens')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, trainable=self.trainable_embeddings,
                                                      pretrained_embeddings=self.pretrained_embeddings)
            input_embd = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            
        with tf.variable_scope('GRU_encoder'):
            _, final_state = layers.GRU_encoder(input_embd, self.hidden_size, dropout_keep_prob=self.dropout_keep_prob,
                                                input_lengths=self.input_lens, n_layers=self.n_layers,
                                                bidirectional=self.bidirectional)
            
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(final_state, self.n_classes, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='output_layer')
            
class GRU_Attn_Net:
    def __init__(self, n_classes, vocab_size, max_sent_len, embedding_dim=100, hidden_size=128, attn_size=128, n_layers=1,
                 bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.max_sent_len = max_sent_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.pretrained_embeddings = pretrained_embeddings
        self.trainable_embeddings = trainable_embeddings
        
        self.inputs = tf.placeholder(tf.int32, [None, self.max_sent_len], name='inputs')
        self.input_lens = tf.placeholder(tf.int32, [None], name='input_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, trainable=self.trainable_embeddings,
                                                      pretrained_embeddings=self.pretrained_embeddings)
            input_embd = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            
        with tf.variable_scope('GRU_attn_encoder'):
            self.final_state = layers.GRU_attn_encoder(input_embd, self.hidden_size, self.attn_size,
                                                       input_lengths=self.input_lens, n_layers=self.n_layers,
                                                       bidirectional=self.bidirectional,
                                                       dropout_keep_prob=self.dropout_keep_prob)
            
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.final_state, self.n_classes, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='output_layer')
            
class HAN_Net:
    def __init__(self, n_classes, vocab_size, n_words, n_sents, embedding_dim=100, hidden_size=128, attn_size=128,
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.n_words = n_words
        self.n_sents = n_sents
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.pretrained_embedings = pretrained_embeddings
        self.trainable_embeddings = trainable_embeddings
        
        self.inputs = tf.placeholder(tf.int32, [None, self.n_sents, self.n_words], name='inputs')
        self.sentence_lens = tf.placeholder(tf.int32, [None, self.n_sents], name='sentence_lens')
        self.doc_lens = tf.placeholder(tf.int32 [None], name='doc_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, trainable=self.trainable_embeddings,
                                                      pretrained_embeddings=self.pretrained_embeddings)
            sentence_list = tf.unstack(self.inputs, axis=1)
            sent_lens_list = tf.unstack(self.sentence_lens, axis=1)
            sent_embd_list = [tf.nn.embedding_lookup(self.embeddings, sentence) for sentence in sentence_list]
            
        with tf.variable_scope('hierarchical_attn_encoder'):
            doc_vector = layers.hierarchical_att_encoder(sent_embd_list, self.hidden_size, self.attn_size,
                                                         sent_lens_list, self.doc_lens, n_layers=self.n_layers,
                                                         bidirectional=self.bidirectional, 
                                                         dropout_keep_prob=self.dropout_keep_prob)
            
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(doc_vector, self.n_classes, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='output_layer')
            
class CNN_Net:
    def __init__(self, n_classes, vocab_size, max_sent_len, layers, layer_filters, embedding_dim,
                 pretrainted_embeddings=None, trainable_embeddings=True, l2_reg_lambda=0.0, train_emb=False, max_pool_size=4):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.max_sent_len = max_sent_len
        self.layers = layers
        self.layer_filters = layer_filters
        self.embedding_dim = embedding_dim
        self.pretrained_embeddings = pretrained_embeddings
        self.trainable_embeddings = trainable_embeddings
        self.l2_reg_lambda = l2_reg_lambda
        self.train_emb = train_emb
        self.max_pool_size = max_pool_size
        
        self.inputs = tf.placeholder(tf.int32, [None, self.max_sent_len], name='inputs')
        self.input_lens = tf.placeholder(tf.int32, [None], name='input_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, trainable=self.trainable_embeddings,
                                                      pretrained_embeddings=self.pretrained_embeddings)
            input_embd = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            
        with tf.variable_scope('cnn_encoder'):
            conv_output = input_embd
            
            for i, layer in enumerate(self.layers):
                filter_outputs = []
                in_channel = int(conv_output.shape[-1])
                for j, filter_size in enumerate(layer):
                    num_filters = self.layer_filters[i][j]
                    
                    conv = layers.convolution_1d(conv_output, 3, in_channel, num_filters, stride=1, padding='SAME',
                                                 name='layer_{}_filter_{}'.format(i, j))
                    relu = tf.nn.relu(conv)
                    max_pool = tf.nn.max_pool(tf.expand_dims(relu, 1), [1, 1, 2, 1], [1, 1, 2, 1], 'SAME')
                    dropout = tf.nn.dropout(max_pool, self.dropout_keep_prob)
                    filter_ouputs.append(dropout)
                conv_output = tf.squeeze(tf.concat(filter_outputs, 3), 1)
            flattened = tf.reshape(conv_output, [-1, int(conv_output.shape[1]*conv_output.shape[2])])
        
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(flattened, self.n_classes, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='output')

class MatchGRU_Net:
    def __init__(self, vocab_size, max_context_len, max_query_len, embedding_dim=100, hidden_size=128,
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True):
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.max_query_len = max_query_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.pretrained_embeddings = pretrained_embeddings
        self.trainable_embeddings = trainable_embeddings
        
        self.context = tf.placeholder(tf.int32, [None, self.max_context_len], name='context')
        self.query = tf.placeholder(tf.int32, [None, self.max_query_len], name='query')
        self.context_lens = tf.placeholder(tf.int32, [None], name='context_lens')
        self.query_lens = tf.placeholder(tf.int32, [None], name='query_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        self.__build_model()
        
    def __build_model(self):
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layers(self.vocab_size, self.embedding_dim, trainable=self.trainable_embeddings,
                                                       pretrained_embeddings=self.pretrained_embeddings)
            context_embd = tf.nn.embedding_lookup(self.embeddings, self.context)
            query_embd = tf.nn.embedding_lookup(self.embeddings, self.query)
            
        with tf.variable_scope('GRU_encoder'):
            c_outputs, c_final_state = layers.GRU_encoder(context_embd, self.hidden_size,
                                                          dropout_keep_prob=self.dropout_keep_prob, 
                                                          input_lengths=self.context_lens,
                                                          n_layers=self.n_layers,
                                                          bidirectional=self.bidirectional)
            tf.get_variable_scope().reuse_variables()
            q_outputs, q_final_state = layers.GRU_encoder(query_embd, self.hidden_size,
                                                          dropout_keep_prob=self.dropout_keep_prob,
                                                          input_lengths=self.query_lens,
                                                          n_layers=self.n_layers,
                                                          bidirectional=self.bidirectional)
            
        with tf.variable_scope('match_GRU'):
            match_outputs, match_final_state = layers.match_GRU(c_outputs, q_outputs, self.hidden_size,
                                                                dropout_keep_prob=self.dropout_keep_prob)
            
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(match_final_state, 1, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='output_layer')