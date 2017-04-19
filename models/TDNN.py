import sys
import os
import numpy as np
import tensorflow as tf
import re
from base import Model

from batch_loader import BatchLoader

layers = tf.contrib.layers

class TDNN(Model):

    def __init__(self, sess, config):
        
        self.sess = sess
        self.config = config
        self.tb_dir = "/data/tensorboard_log_dict/TDNN_white"
        self.loader = BatchLoader(self.config)
        
        

    def build_model(self, input_):
        with tf.variable_scope("TDNN"):
            with tf.variable_scope("conv") as scope:
                maps = []

                for idx, kernel_dim in enumerate(self.config.kernels):
                    #if idx < 3:
                    #    gpu_num = 0
                    #else:
                        #gpu_num = idx-2
                    #    gpu_num = 1
                    #with tf.device('/gpu:%d' % gpu_num):
                    reduced_length = input_.get_shape()[1] - kernel_dim + 1

                    # [batch_size x seq_length x embed_dim x feature_map_dim]
                    conv = layers.conv2d(input_, self.config.feature_maps[idx], [kernel_dim, self.config.binary_embed_width], 1, padding = 'VALID',
                                             scope = 'conv'+str(idx), weights_initializer = layers.xavier_initializer_conv2d())

                    # [batch_size x 1 x 1 x feature_map_dim]
                    pool = layers.max_pool2d(conv, [reduced_length,1], 1, scope='pool'+str(idx))

                    maps.append(tf.squeeze(pool))

                fc = tf.concat(axis=1, values=maps)
                
            with tf.variable_scope("fully"):
                
                fc = tf.nn.dropout(fc, self.config.dropout_prob)

                flat = tf.reshape(fc, [self.config.batch_size, sum(self.config.feature_maps)])
                
                
                
                
                fc1 = layers.fully_connected(flat, 2048, activation_fn = tf.nn.relu, scope='fc1',
                                             weights_initializer = layers.xavier_initializer(),
                                             biases_initializer = tf.constant_initializer(0.01))
                
                fc1 = tf.contrib.layers.batch_norm(fc1, decay=0.9, center=True, scale=True, epsilon=True, activation_fn=tf.nn.relu)
                
                
                fc2 = layers.fully_connected(fc1, 1024, activation_fn = tf.nn.relu, scope='fc2',
                                             weights_initializer = layers.xavier_initializer(),
                                             biases_initializer = tf.constant_initializer(0.01))
                
                fc2 = tf.contrib.layers.batch_norm(fc2, decay=0.9, center=True, scale=True, epsilon=True, activation_fn=tf.nn.relu)
                
                logits = layers.fully_connected(fc2, self.config.num_classes, activation_fn = None, scope='logits',
                                             weights_initializer = layers.xavier_initializer(),
                                             biases_initializer = tf.constant_initializer(0.01))
             
            return logits
        
    def build_loss(self, logits, labels):
        weight_decay_rate = 0.0001

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
        tf.summary.scalar('loss_', cross_entropy_mean)

        weights_only = filter( lambda x: x.name.endswith('w:0'), tf.trainable_variables() )
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
        cross_entropy_mean += weight_decay

        return cross_entropy_mean
    

          
    def inference(self, input_):
        self.loader = BatchLoader(self.config.data_dir, self.config.dataset_name, self.config.batch_size, self.config.num_classes, self.config.preprocessor, self.config.epoch, self.config.specialist, self.config.forward_only)

        
        
        content, filename = self.loader.prepare_inference()
        
        #with tf.control_dependencies([self.loader.enqueue]):
        logits = self.build_model(content)
        softmax = tf.nn.softmax(logits)
        
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        self.sess.run(init_op)

        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('no checkpoint found...')
        
        self.sess.run(self.loader.enqueue, feed_dict={self.loader.filenames:input_})
        
        m_logits, m_softmax, m_filename = self.sess.run([logits, softmax, filename])
        
        print(m_softmax, m_filename)
        
        

            
    def tower_loss(self, scope, content, label, gpu_num):
        
        #self.label = label
        #self.path = path
        
        logits = self.build_model(content)
        loss = self.build_loss(logits, label)
        
        self.accuracy = self.build_accuracy(logits, label)
        
        tf.add_to_collection('losses', loss)
        
        losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(losses, name='total_loss')
        
        for l in losses + [total_loss]:
            loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
        
        return total_loss
    
    def average_gradients(self, tower_grads):
  
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
    
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)

                grads.append(expanded_g)

            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
            
        return average_grads
    
    
    def run(self):
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
        if not self.config.forward_only:
        
            self.opt = tf.train.AdamOptimizer(self.config.learning_rate)
            tower_grads = []
            
            content, label = self.loader.data_type_dict[self.config.data_type](self.config.forward_only)
            self.label = label
            
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(2):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            self.loss = self.tower_loss(scope, content, label, i)

                            tf.get_variable_scope().reuse_variables()

                            grads = self.opt.compute_gradients(self.loss)

                            tower_grads.append(grads)

            grads = self.average_gradients(tower_grads)
            
            

            apply_gradient_op = self.opt.apply_gradients(grads, global_step=self.global_step)

            self.train_op = apply_gradient_op
        
        else:
            content, label = self.loader.data_type_dict[self.config.data_type](self.config.forward_only)
            self.label = label
            logits = self.build_model(content)
            self.accuracy = self.build_accuracy(logits, label)
        
        # ready for train
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        self.sess.run(init_op)
        
        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("/data/tensorboard_log/dict/TDNN_0404/", self.sess.graph)
        
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)
        
        ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('no checkpoint found...')
        

        if not self.config.forward_only:
            self.train()
            #self.get_logits()
            print("train")
        else:
            self.test()
            #self.get_logits()