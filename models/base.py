import os
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from tensorflow.contrib.tensorboard.plugins import projector

class Model(object):
    
    def __init__(self):
        
        print("init base model")
  
    def build_accuracy(self, logits, labels):
        correct_prediction = tf.nn.in_top_k(logits, labels, 1)
        self.true_label = tf.argmax(logits, axis=1)
        self.logits = logits
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.summary.scalar('accuarcy', accuracy, 1)
        return accuracy
        

    def train(self):
        try:
            while not self.coord.should_stop():
                _,  loss, acc, merged, global_step, label, true_label, logits = self.sess.run([self.train_op, self.loss, self.accuracy, self.merged_summary, self.global_step, self.label, self.true_label, self.logits])
                if global_step % 10 == 0:
                    print('step : %g, loss : %g, accuracy : %g'%(global_step, loss, acc))
                    print(label)
                    print(true_label)
                    print(logits)
                if global_step % self.config.summary_iter == 0:
                    self.writer.add_summary(merged, global_step)
                    print('write summary')
                if global_step % self.config.save_iter == 0:
                    self.saver.save(self.sess, os.path.join(self.config.checkpoint_dir,'model.ckpt'), global_step)
                    print('save checkpoint')

        except tf.errors.OutOfRangeError as e:
            print('end of epochs', e)

        except Exception as e:
            print('error occurred', e)

        finally:
            
            self.coord.join(self.threads)
            self.coord.should_stop()
            self.coord.clear_stop()            
        
        
    def test(self):
        valacc = []
        step = 0
        run_metadata = tf.RunMetadata()
        logit_list = []
        label_list = []
        try:
            #while step < 100:
            while not self.coord.should_stop():
                acc, global_step, label, true_label, logits = self.sess.run([self.accuracy,self.global_step, self.label, self.true_label, self.logits]
                                                 ,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                                 ,run_metadata=run_metadata
                                                )
                for lo in logits:
                    logit_list.append(lo)
                    
                for la in label:
                    label_list.append(la)
                """
                
                if step % 10 == 0:
                    trace_file = open('tracing/local_8/timeline_%s.json' % str(step), 'wb')
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)

                    trace_file.write(trace.generate_chrome_trace_format())
                    trace_file.close()
                    
                """
                valacc.append(acc)
                step += 1
                print('step : %g, accuracy : %g'%(step, acc))
                print(label)
                print(true_label)
                
                #if step < 2:
                
                    #break
                #print(trace.generate_chrome_trace_format())
                
                if step % 100 == 0:
                    print('total acc : %g'%(np.mean(valacc)))
            
            
            
        except tf.errors.OutOfRangeError as e:
            print('len : ', len(valacc))
            print('eval accuracy : ',np.mean(valacc))
            print('end of epochs', e)
            

        except Exception as e:
            print('error occurred', e)

        finally:
            ll = np.array(logit_list)
            #print(ll.shape)
            logit_tensor = tf.Variable(ll, name='bottleneck')
            init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
            self.sess.run(init_op)
            
            self.generate_metadata_file(self.sess, label_list)
            self.generate_embeddings(self.sess, logit_tensor.name)
            self.coord.join(self.threads)
            self.coord.should_stop()
            self.coord.clear_stop()
            
            
    
    def generate_embeddings(self, sess, tensor_name):
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.tb_dir, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = tensor_name
        embed.metadata_path = os.path.join(self.tb_dir, 'metadata.tsv')

        projector.visualize_embeddings(writer, config)

        saver.save(sess, os.path.join(self.tb_dir,'model.ckpt'))


    def generate_metadata_file(self, sess, label_list):
        with open(os.path.join(self.tb_dir,'metadata.tsv'), 'w') as f:
            for v in label_list:
                f.write('{}\n'.format(v))      