import numpy as np
import os
import tensorflow as tf
from config import Config
from models import *


flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train [10]")
flags.DEFINE_integer("binary_embed_width", 32, "The width of binary embedding matrix [32]")
flags.DEFINE_integer("binary_embed_height", 12000, "The height of binary embedding matrix [12000]")
flags.DEFINE_integer("batch_size", 50, "The size of batch images [50]")
flags.DEFINE_integer("classes_num", 25, "Classes number [25]")
flags.DEFINE_integer("summary_iter", 100, "Iteration of summay write [100]")
flags.DEFINE_integer("save_iter", 1000, "Iteration of save checkpoint [1000]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate [0.0001]")
flags.DEFINE_float("decay", 0.5, "Decay of SGD [0.5]")
flags.DEFINE_float("dropout_prob", 0.5, "Probability of dropout layer [0.5]")
flags.DEFINE_string("feature_maps", "[100,150,200,250,300]", "The # of feature maps in CNN [100,200,300,400,500]")
flags.DEFINE_string("kernels", "[1,2,3,4,5]", "The width of CNN kernels [1,2,3,4,5]")
flags.DEFINE_string("model", "Resnet", "The type of model to train and test [TDNN]")
flags.DEFINE_string("preprocessor", "thumbnail", "The type of preprocessor [thumbnail]")
flags.DEFINE_string("data_type", "tfrecord", "The type of data [tfrecord]")
flags.DEFINE_string("data_dir", "/data/dict/PE", "The name of data directory [data]")
flags.DEFINE_string("dataset", "raw", "data set [raw]")
flags.DEFINE_string("checkpoint_dir", "/data/dict/refactoring/checkpoint_dist/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("GPU", "0,1,2,3", "The name of data directory [data]")
flags.DEFINE_string("label_query", "select * from pe_list where num > 9", "label_query [data]")
flags.DEFINE_string("data_query", "select * from win32_train_list_clone order by md5", "data_query [data]")
flags.DEFINE_boolean("forward_only", False, "True for forward only, False for training [False]")
flags.DEFINE_boolean("specialist", False, "True for specialist, False for generalist [False]")
flags.DEFINE_boolean("whitelist", True, "True for whitelist, False for generalist [False]")

FLAGS = flags.FLAGS


model_dict = {
    'LSTM': None,
    'LSTMCNN': None,
    'TDNN': TDNN,
    'Resnet': Resnet,
}



def main(_):
    print(flags.FLAGS.__flags)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU
    
    
    config = Config(epoch=FLAGS.epoch,
                    binary_embed_width=FLAGS.binary_embed_width,
                    binary_embed_height=FLAGS.binary_embed_height,
                    feature_maps=eval(FLAGS.feature_maps),
                    kernels=eval(FLAGS.kernels),
                    batch_size=FLAGS.batch_size,
                    classes_num=FLAGS.classes_num,
                    dropout_prob=FLAGS.dropout_prob,
                    forward_only=FLAGS.forward_only,
                    dataset_name=FLAGS.dataset,
                    data_dir=FLAGS.data_dir,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    learning_rate=FLAGS.learning_rate,
                    decay=FLAGS.decay,
                    preprocessor=FLAGS.preprocessor,
                    specialist=FLAGS.specialist,
                    whitelist=FLAGS.whitelist,
                    data_type=FLAGS.data_type,
                    summary_iter=FLAGS.summary_iter,
                    save_iter=FLAGS.save_iter,
                    label_query=FLAGS.label_query,
                    data_query=FLAGS.data_query
                   )

    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        model = model_dict[FLAGS.model](sess, config)

        
        model.run()
        #export(model)
if __name__ == '__main__':
    tf.app.run()
