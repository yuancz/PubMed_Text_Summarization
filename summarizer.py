import tensorflow as tf
import functions

class Text_Summarization:
    def __init__(self, net, lr=0.001, mode='train'):
        self.lr = lr
        
        self._enc_inputs = net.enc_inputs
        self._dec_inputs = net.dec_inputs
        self._enc_lens = net.enc_lens
        self._dec_lens = net.dec_lens
        self._dropout_keep_prob = net.dropout_keep_prob
        self._teacher_forcing = net.teacher_forcing
        self._teacher_forcing_mask = net.teacher_forcing_mask
        self._logits = net.logits
        self._is_training = net.is_training
        self._sample_decoding = net.sample_decoding
        
        self._targets = tf.placeholder(tf.int32, [None, net.max_dec_len], name='targets')
        self._target_lens = tf.placeholder(tf.int32, [None], name='target_lens')
        self._batch_size = tf.placeholder(tf.int32, name='batch_size')
        self._loss_mask_len = tf.placeholder(tf.int32, shape=[1], name='loss_mask_len')
        self._lr = tf.placeholder(tf.float32, name='lr')
        
        assert mode in ['train', 'eval', 'infer']
        if mode == 'train':
            self._build_eval_metrics()
            self._build_optimizer()
        elif mode == 'eval':
            self._build_loss()
            self._build_eval_metrics()
        elif mode == 'infer':
            self._build_predictions()
    
    def _build_predictions(self):
        self._softmax = tf.nn.softmax(self._logits)
        self._predictions = tf.to_int32(tf.argmax(self._softmax, axis=2))
        
    def _build_loss(self):
        self._build_predictions()
        loss_mask = tf.minimum(self._loss_mask_len, self._target_lens)
        self._loss = functions.masked_log_loss(self._softmax, self._targets, loss_mask, self._batch_size)
        
    def _build_eval_metrics(self):
        self._build_predictions()
        
        self._accuracy = functions.compute_masked_accuracy(self._predictions, self._targets, self._target_lens)
        
    def _build_optimizer(self):
        self._build_loss()
        
        params = tf.trainable_variables()
        #gradients = tf.gradients(self._loss, params, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self._loss, params), 1) 
        self._gradient_norm = tf.global_norm(gradients)
        
        opt_func = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._optimizer = opt_func.apply_gradients(zip(gradients, params))
        #self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self._loss) 
        
        
    #def load_weights(self, sess, checkpoint_dir, vars_list=None):
        
    def train_step(self, sess, enc_inputs, dec_inputs, enc_lens, dec_lens, targets, target_lens, dropout_keep_prob=1.0,
                   teacher_forcing=True, teacher_forcing_mask=None, loss_mask_len=None, sample_decoding=False, lr=None):
        if teacher_forcing_mask is None:
            teacher_forcing_mask = [1 for _ in range(len(targets[0]))]
        if lr is None:
            lr = self.lr
        if loss_mask_len is None:
            loss_mask_len = [len(targets[0])]
        else:
            loss_mask_len = [loss_mask_len]
        batch_size = len(enc_inputs)
        feed_dict = {self._enc_inputs : enc_inputs,
                     self._dec_inputs : dec_inputs,
                     self._enc_lens : enc_lens,
                     self._dec_lens : dec_lens,
                     self._targets : targets,
                     self._target_lens : target_lens,
                     self._dropout_keep_prob : dropout_keep_prob,
                     self._teacher_forcing : teacher_forcing,
                     self._teacher_forcing_mask : teacher_forcing_mask,
                     self._batch_size : batch_size,
                     self._loss_mask_len : loss_mask_len, 
                     self._is_training : True,
                     self._sample_decoding : sample_decoding,
                     self._lr : lr}
        run_vars = [self._loss, self._accuracy, self._gradient_norm, self._optimizer]
        
        loss, accuracy, grad_norm, _ = sess.run(run_vars, feed_dict=feed_dict)
        return loss, accuracy, grad_norm
        
    def val_step(self, sess, enc_inputs, dec_inputs, enc_lens, dec_lens, targets, target_lens, sample_decoding=False):
        batch_size = len(enc_inputs)
        feed_dict = {self._enc_inputs : enc_inputs,
                     self._dec_inputs : dec_inputs,
                     self._enc_lens : enc_lens,
                     self._dec_lens : dec_lens,
                     self._targets : targets,
                     self._target_lens : target_lens,
                     self._dropout_keep_prob : 1.0,
                     self._teacher_forcing : False,
                     self._teacher_forcing_mask : [1 for _ in range(len(targets[0]))],
                     self._batch_size : batch_size,
                     self._loss_mask_len : [len(targets[0])],
                     self._is_training : False,
                     self._sample_decoding : sample_decoding}
        run_vars = [self._loss, self._accuracy]
        
        loss, accuracy = sess.run(run_vars, feed_dict=feed_dict)
        return loss, accuracy
        
    def deploy(self, sess, enc_inputs, enc_lens, dummy_dec_inputs, sample_decoding=False):
        batch_size = len(enc_inputs)
        feed_dict = {self._enc_inputs : enc_inputs,
                     self._dec_inputs : dummy_dec_inputs,
                     self._enc_lens : enc_lens,
                     self._dropout_keep_prob : 1.0,
                     self._teacher_forcing : False,
                     self._teacher_forcing_mask : [1 for _ in range(len(dummy_dec_inputs[0]))],
                     self._batch_size : batch_size, 
                     self._is_training : False,
                     self._sample_decoding : sample_decoding}
        run_vars = self._predictions
        
        predictions = sess.run(run_vars, feed_dict=feed_dict)
        return predictions
    
class Multitask_LM_Summarization:
    def __init__(self, net, lr=0.001, mode='train', task='all'):
        assert(task in ['all', 'summarize', 'language'])
        self.lr = lr
        self.task = task
        self._enc_inputs = net.enc_inputs
        self._dec_inputs = net.dec_inputs
        self._enc_lens = net.enc_lens
        self._dec_lens = net.dec_lens
        self._dropout_keep_prob = net.dropout_keep_prob
        self._teacher_forcing = net.teacher_forcing
        self._logits = net.logits
        self._lm_logits = net.lm_logits
        self._seed_length = net.seed_length
        
        self._targets = tf.placeholder(tf.int32, [None, net.max_dec_len], name='targets')
        self._target_lens = tf.placeholder(tf.int32, [None], name='target_lens')
        self._lm_targets = tf.stack(tf.unstack(self._enc_inputs, axis=1)[1:], axis=1)
        self._batch_size = tf.placeholder(tf.int32, name='batch_size')
        self._sum_loss_weight = tf.placeholder(tf.float32, name='sum_loss_weight')
        self._lm_loss_weight = tf.placeholder(tf.float32, name='lm_loss_weight')
        
        assert mode in ['train', 'eval', 'infer']
        if mode == 'train':
            self._build_eval_metrics()
            self._build_optimizer()
        elif mode == 'eval':
            self._build_loss()
            self._build_eval_metrics()
        elif mode == 'infer':
            self._build_predictions()
    
    def _build_predictions(self):
        self._softmax = tf.nn.softmax(self._logits)
        self._lm_softmax = tf.stack(tf.unstack(tf.nn.softmax(self._lm_logits), axis=1)[:-1], axis=1)
        self._predictions = tf.to_int32(tf.argmax(self._softmax, axis=2))
        self._lm_predictions = tf.to_int32(tf.argmax(self._lm_softmax, axis=2))
        
    def _build_loss(self):
        self._build_predictions()
        if self.task == 'all':
            self._lm_loss = functions.masked_log_loss(self._lm_softmax, self._lm_targets, self._enc_lens, self._batch_size)
            self._sum_loss = functions.masked_log_loss(self._softmax, self._targets, self._target_lens, self._batch_size)
        elif self.task == 'summarize':
            self._lm_loss = tf.constant(0.0)
            self._sum_loss = functions.masked_log_loss(self._softmax, self._targets, self._target_lens, self._batch_size)
        elif self.task == 'language':
            self._lm_loss = functions.masked_log_loss(self._lm_softmax, self._lm_targets, self._enc_lens, self._batch_size)
            self._sum_loss = tf.constant(0.0)
        
        self._loss = self._lm_loss_weight*self._lm_loss + self._sum_loss_weight*self._sum_loss
        
    def _build_eval_metrics(self):
        self._build_predictions()
        self._accuracy = functions.compute_masked_accuracy(self._predictions, self._targets, self._target_lens)
        self._lm_accuracy = functions.compute_masked_accuracy(self._lm_predictions, self._lm_targets, self._enc_lens)
        
    def _build_optimizer(self):
        self._build_loss()
        
        params = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self._loss, params), 1) 
        self._gradient_norm = tf.global_norm(gradients)

        opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._optimizer = opt_func.apply_gradients(zip(gradients, params))
        
    #def load_weights(self, sess, checkpoint_dir, vars_list=None):
        
    def train_step(self, sess, enc_inputs, dec_inputs, enc_lens, dec_lens, targets, target_lens, dropout_keep_prob=1.0,
                   teacher_forcing=True, seed_length=None, sum_loss_weight=0.5, lm_loss_weight=0.5):
        if seed_length is None:
            seed_length = len(enc_inputs[0])
        batch_size = len(enc_inputs)
        feed_dict = {self._enc_inputs : enc_inputs,
                     self._dec_inputs : dec_inputs,
                     self._enc_lens : enc_lens,
                     self._dec_lens : dec_lens,
                     self._targets : targets,
                     self._target_lens : target_lens,
                     self._dropout_keep_prob : dropout_keep_prob,
                     self._teacher_forcing : teacher_forcing,
                     self._batch_size : batch_size,
                     self._seed_length : seed_length,
                     self._sum_loss_weight : sum_loss_weight,
                     self._lm_loss_weight : lm_loss_weight}
        run_vars = [self._loss, self._sum_loss, self._lm_loss, self._accuracy, self._lm_accuracy, 
                    self._gradient_norm, self._optimizer]
        
        loss, sum_loss, lm_loss, accuracy, lm_accuracy, grad_norm, _ = sess.run(run_vars, feed_dict=feed_dict)
        return loss, sum_loss, lm_loss, accuracy, lm_accuracy, grad_norm
        
    def val_step(self, sess, enc_inputs, dec_inputs, enc_lens, dec_lens, targets, target_lens, seed_length=None,
                 sum_loss_weight=0.5, lm_loss_weight=0.5):
        if seed_length is None:
            seed_length = len(enc_inputs[0])
        batch_size = len(enc_inputs)
        feed_dict = {self._enc_inputs : enc_inputs,
                     self._dec_inputs : dec_inputs,
                     self._enc_lens : enc_lens,
                     self._dec_lens : dec_lens,
                     self._targets : targets,
                     self._target_lens : target_lens,
                     self._dropout_keep_prob : 1.0,
                     self._teacher_forcing : False,
                     self._batch_size : batch_size,
                     self._seed_length : seed_length,
                     self._sum_loss_weight : sum_loss_weight,
                     self._lm_loss_weight : lm_loss_weight}
        run_vars = [self._loss, self._sum_loss, self._lm_loss, self._accuracy, self._lm_accuracy]
        
        loss, sum_loss, lm_loss, accuracy, lm_accuracy = sess.run(run_vars, feed_dict=feed_dict)
        return loss, sum_loss, lm_loss, accuracy, lm_accuracy
        
    def deploy(self, sess, enc_inputs, enc_lens, dummy_dec_inputs, seed_length=None):
        if seed_length is None:
            seed_length = len(enc_inputs[0])
        batch_size = len(enc_inputs)
        feed_dict = {self._enc_inputs : enc_inputs,
                     self._dec_inputs : dummy_dec_inputs,
                     self._enc_lens : enc_lens,
                     self._dropout_keep_prob : 1.0,
                     self._teacher_forcing : False,
                     self._batch_size : batch_size, 
                     self._seed_length : seed_length}
        run_vars = [self._predictions, self._lm_predictions]
        
        predictions, lm_predictions = sess.run(run_vars, feed_dict=feed_dict)
        return predictions, lm_predictions