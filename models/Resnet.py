import sys
import os
import numpy as np
import tensorflow as tf
import re


from base import Model

from batch_loader import BatchLoader
from resnet_config import Config
from resnet_utils import *
from resnet_utils import _max_pool

layers = tf.contrib.layers

class Resnet(Model):

    def __init__(self, sess, config):
        
        self.sess = sess
        self.config = config
        
        self.loader = BatchLoader(self.config)
            
                
    def _inference(self, x, is_training,
              num_classes=25,
              num_blocks=[3, 4, 6, 3],
              use_bias=False,
              bottleneck=True):
        c = Config()
        c['bottleneck'] = bottleneck
        c['is_training'] = tf.convert_to_tensor(is_training,
                                                dtype='bool',
                                                name='is_training')
        c['ksize'] = 3
        c['stride'] = 1
        c['use_bias'] = use_bias
        c['fc_units_out'] = num_classes
        c['num_blocks'] = num_blocks
        c['stack_stride'] = 2

        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 64
            c['ksize'] = 7
            c['stride'] = 2
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('scale2'):
            x = _max_pool(x, ksize=3, stride=2)
            c['num_blocks'] = num_blocks[0]
            c['stack_stride'] = 1
            c['block_filters_internal'] = 64
            x = stack(x, c)

        with tf.variable_scope('scale3'):
            c['num_blocks'] = num_blocks[1]
            c['block_filters_internal'] = 128
            assert c['stack_stride'] == 2
            x = stack(x, c)

        with tf.variable_scope('scale4'):
            c['num_blocks'] = num_blocks[2]
            c['block_filters_internal'] = 256
            x = stack(x, c)

        with tf.variable_scope('scale5'):
            c['num_blocks'] = num_blocks[3]
            c['block_filters_internal'] = 512
            x = stack(x, c)

        x = tf.reduce_mean(x, axis=[1, 2], name="avg_pool")

        if num_classes != None:
            with tf.variable_scope('fc'):
                x = fc(x, c)

        return x
    
    def build_model(self, input_):
        with tf.variable_scope("Resnet"):
            
            logits = self._inference(input_,
                                    num_classes=self.config.num_classes,
                                    is_training=True,
                                    bottleneck=False,
                                    num_blocks=[2, 2, 2, 2])
            return logits
        
    def build_loss(self, logits, labels):
        #weight_decay_rate = 0.001
        """
        Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
        #tf.summary.scalar('loss', cross_entropy_mean)

        #weights_only = filter( lambda x: x.name.endswith('w:0'), tf.trainable_variables() )
        #weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
        #cross_entropy_mean += weight_decay

        return cross_entropy_mean
    
    def build_accuracy(self, logits, labels):
        correct_prediction = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.summary.scalar('accuarcy', accuracy)
        return accuracy
        

    """
    def get_logits(self):
        import pymysql
        conn = pymysql.connect(host='172.17.0.2', port=3306, user='root', password='deep.est', db='vato', autocommit=True)
        curs = conn.cursor()
        
        valacc = []
        vallogit = []
        vallabel = []
        valpath = []
        
        #sql = "insert into vato.m_logits(md5,logits_gen) values (%s, %s)"
        sql = "update vato.m_logits as t set t.logits_sp2 = %s where t.md5 = %s"    
        
        try:
            while not self.coord.should_stop():
                logit, label, path, loss, acc, global_step = self.sess.run([self.logits, self.label, self.path, self.loss, self.accuracy, self.global_step])
                
                
                for idx in range(len(logit)):
                    #curs.execute(sql, (path[idx], str(logit[idx])))
                    curs.execute(sql, (str(logit[idx]), path[idx]))
                
                valacc.append(acc)
                
                print('loss : %g, accuracy : %g'%(loss, acc))
                

        except tf.errors.OutOfRangeError as e:
            print('len : ', len(valpath))
            print('eval accuracy : ',np.mean(valacc))
            
            
            
            #conn.commit()
            conn.close()
            print('end of epochs', e)
            

        except Exception as e:
            print('error occurred', e)

        finally:
            self.coord.join(self.threads)
            self.coord.should_stop()
            self.coord.clear_stop()
    """
    def inference(self, input_):
        

        
        
        content, filename = self.loader.prepare_inference()
        
        self.result_path = filename
        #with tf.control_dependencies([self.loader.enqueue]):
        logits = self.build_model(content)
        softmax = tf.nn.softmax(logits)
        
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        self.sess.run(init_op)

        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
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
            
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(4):
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
            logits = self.build_model(content)
            self.accuracy = self.build_accuracy(logits, label)
        
        # ready for train
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        self.sess.run(init_op)
        
        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("/data/tensorboard_log/dict/Resnet_0404/", self.sess.graph)
        
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