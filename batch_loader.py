import re
import os
import math
import numpy as np
import csv
import tensorflow as tf
from DBLoader import DBLoader
from parallel_reader import ParallelReader

class BatchLoader(object):

    def __init__(self, config):
        
        self.num_epoch = config.epoch
        
        self.preprocessor_dict = {'thumbnail' : self.prepare_thumbnail,
                                  'sequence' : self.prepare_sequence,
                                  'simhash' : self.prepare_simhash,}
        
        self.data_type_dict = {'db' : self.dataset_from_db,
                               'tfrecord' : self.dataset_from_record,}
        
        self.read_size = config.binary_embed_width * config.binary_embed_height
        self.specialist = config.specialist
        self.preprocessor = self.preprocessor_dict[config.preprocessor]
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.forward_only = config.forward_only
        self.config = config
        
        self.features = {
            #'label': tf.FixedLenFeature([], tf.int64),
            #'image_raw': tf.FixedLenFeature([], tf.string)}
            #'image_raw': tf.FixedLenFeature([224*224], tf.float32)}
            'image_raw': tf.VarLenFeature(tf.float32),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)}

        self.label_feature = {
            'label': tf.FixedLenFeature([], tf.int64)
        }
        
        

        
    def get_label(self, filenames):
        array = np.array([self.class_names_to_ids[label_string.split('/')[-2]] for label_string in filenames], dtype=np.int64)
        
        """ specialist_1
        for idx in range(len(array)):
            if array[idx] == self.class_names_to_ids["Downloader"]:
                array[idx] = 0
            elif array[idx] == self.class_names_to_ids["Virut"]:
                array[idx] = 1
            elif array[idx] == self.class_names_to_ids["MSIL"]:
                array[idx] = 2
            else:
                array[idx] = 3
        """
        
        """ specialist_2
        
        for idx in range(len(array)):
            if array[idx] == self.class_names_to_ids["Bundler"]:
                array[idx] = 0
            elif array[idx] == self.class_names_to_ids["Virut"]:
                array[idx] = 1
            elif array[idx] == self.class_names_to_ids["LoadMoney"]:
                array[idx] = 2
            else:
                array[idx] = 3
        """
        
        return array
    
    def prepare_thumbnail(self, content):
        content = tf.decode_raw(content, tf.uint8)
        zero_padding_size = tf.constant(224) - tf.mod(tf.shape(content), tf.constant(224))
        zero_padding = tf.zeros(zero_padding_size, dtype=tf.uint8)
        content = tf.concat(axis=0, values=[content, zero_padding])

        content = tf.reshape(content, [-1, 224, 1])
        content = tf.image.resize_images(content, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        content = tf.div(tf.subtract(tf.cast(content, tf.float32), 127.5), 127.5)
        return content
    
    def prepare_sequence(self, content):
        image = tf.decode_raw(content, tf.uint8)
        zero_padding = tf.zeros(tf.nn.relu([2*self.read_size] - tf.shape(image)), dtype=tf.uint8)
        image = tf.concat(axis=0, values=[image, zero_padding])

        image = tf.slice(image, [0], [2*self.read_size])

        image = tf.reshape(image, [2, self.read_size])

        image = tf.matmul(np.array([[0x100, 0x1]], dtype=np.float32), tf.cast(image, dtype=tf.float32))

        image = tf.reshape(image, [self.config.binary_embed_height, self.config.binary_embed_width, 1])


        image = tf.div(tf.cast(image, tf.float32), 65535.0)

        return image
    
    def prepare_simhash(self, image):
        zero_padding = tf.zeros(tf.nn.relu([self.read_size] - tf.shape(image)), dtype=tf.int64)
        image = tf.concat(axis=0, values=[image, zero_padding])
        image = tf.slice(image, [0], [self.read_size])
        
        
        image = tf.reshape(image, [self.config.binary_embed_height, self.config.binary_embed_width, 1])


        image = tf.div(tf.cast(image, tf.float32), 65535.0)

        return image
    
    def dataset_from_record(self, forward_only):
        NUM_PREPROCESS_THREADS = 8
        
        
        if not forward_only:
            record_path = ["/data/dict/tfrecord/tdnn_imascate_train_order_new.tfrecord"]
            filename_queue = tf.train.string_input_producer(record_path, num_epochs=None)
            
            
        else:
            record_path = ["/data/dict/simhash/tdnn_imascate_test_order.tfrecord"]
            filename_queue = tf.train.string_input_producer(record_path, num_epochs=None)
        #ParallelReader
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        
        #train_x, train_y = self.process(serialized_example)
        #common_queue = tf.RandomShuffleQueue(
        #  capacity=1000,
        #  dtypes=[tf.string, tf.string])
        #p_reader = ParallelReader(tf.TFRecordReader, common_queue, num_readers=8)
        
        #_, serialized_example = p_reader.read(filename_queue)
        
        features = tf.parse_single_example(
            serialized_example,
            features={
              'feature': tf.VarLenFeature(tf.float32),
              'label': tf.FixedLenFeature([], tf.int64),
            })
        
        content = tf.sparse_tensor_to_dense(features['feature'])
        
        content = tf.reshape(content, [self.config.binary_embed_height, self.config.binary_embed_width, 1])
        
        train_x = tf.cast(content, tf.float32)
        
        #train_x = self.preprocessor(content)
        train_y = tf.cast(features['label'], tf.int32)
        
        
        batch_set = [train_x, train_y]
        batch_size = self.batch_size
        
        min_after_dequeue = 50
        capacity = min_after_dequeue + 3 * self.batch_size
        if self.config.whitelist:
            tfrecord_train_dir = ["/data/TFRecord/oldvato/Win32 EXE/top1000/notfft/whitelist.tfrecords"]
            whitelist_x = []
            whitelist_y = []
            batch_set = [train_x, train_y]
            #batch_size = int(self.batch_size/2)
            
            t_x, t_y = tf.train.shuffle_batch(batch_set,
                                                        batch_size = batch_size, 
                                                        min_after_dequeue=min_after_dequeue, 
                                                        capacity=capacity, 
                                                        num_threads = NUM_PREPROCESS_THREADS)
            #image_batch = tf.concat(axis=0, values=[t_x, w_x])
            #label_batch = tf.concat(axis=0, values=[t_y, w_y])
            
            return t_x, t_y
            

        
        else:
        
            image_batch, label_batch = tf.train.shuffle_batch(batch_set,
                                                            batch_size = self.batch_size, 
                                                            min_after_dequeue=min_after_dequeue, 
                                                            capacity=capacity, 
                                                            num_threads = NUM_PREPROCESS_THREADS)

            
            #label_batch = tf.Print(label_batch, [label_batch])
            return image_batch, label_batch
    
    def dataset_from_db(self, forward_only):
        
        dbl = DBLoader(db="base")
        
        self.class_names_to_ids = dbl.get_filenames_dict(self.config.label_query)
        
        print(self.class_names_to_ids)
        
        if not self.forward_only:

            sql = self.config.data_query
            #sql = "select * from win32_train_list_clone order by md5"
            #sql = "select t.md5, s.Kaspersky, t.family from sp_list_new as s join win32_train_list_clone as t on s.md5 = t.md5 order by t.md5"

            
            cursor = dbl.sql_request(sql)

            self._train_filenames, self._train_labels = dbl.get_filenames_from_list_temp(cursor, dirpath=self.config.data_dir)
            #self._train_filenames, self._train_labels = dbl.get_filenames_from_list_temp(cursor, dirpath="/data/files/vato_clone/VIRUSCRAWLER/storage/Win32 EXE")
            
            self.train_filenames = []
            self.train_labels = []
            for idx in range(len(self._train_labels)):
                if self._train_labels[idx] in self.class_names_to_ids:
                    self.train_labels.append(self.class_names_to_ids[self._train_labels[idx]])
                    self.train_filenames.append(self._train_filenames[idx])
                
            print("data load done. Number of datasets in train: %d" % (len(self.train_filenames)))
        else:
            sql = self.config.data_query
            #sql = "select * from win32_test_list_clone order by md5"
            cursor = dbl.sql_request(sql)
            self._test_filenames, self._test_labels = dbl.get_filenames_from_list_temp(cursor, dirpath=self.config.data_dir)

            self.test_filenames = []
            self.test_labels = []
            for idx in range(len(self._test_labels)):
                if self._test_labels[idx] in self.class_names_to_ids:
                    self.test_labels.append(self.class_names_to_ids[self._test_labels[idx]])
                    self.test_filenames.append(self._test_filenames[idx])
                
            print("data load done. Number of datasets in test: %d" % (len(self.test_filenames)))

        if self.config.whitelist:
            self.class_names_to_ids = dict(self.class_names_to_ids.items() + {'white':len(self.class_names_to_ids)}.items())
            self.num_classes = len(self.class_names_to_ids)
            
            dbl = DBLoader(db='vato')
            sql = "select path from whitelist order by md5"
            cursor = dbl.sql_request(sql)
            whitepath = []
            for row in cursor:
                whitepath.append(row[0].encode())
                
            bound = int(len(whitepath) * 0.7)
            if not self.forward_only:
                for idx in range(0, bound):
                    self.train_filenames.append(whitepath[idx])
                    self.train_labels.append(self.num_classes-1)
            else:
                for idx in range(bound, len(whitepath)):
                    self.test_filenames.append(whitepath[idx])
                    self.test_labels.append(self.num_classes-1)
        
        self.id_to_name = ["" for x in range(self.num_classes)]
        for k, v in self.class_names_to_ids.iteritems():
            self.id_to_name[v] = k
        
        dbl.close_conn()
        
        return self.prepare_dataset(forward_only)
        
    
    def prepare_dataset(self, forward_only):
        NUM_PREPROCESS_THREADS = 8
        if not forward_only:
            filenames = self.train_filenames
            labels = self.train_labels
            filename_queue = tf.train.input_producer(filenames, num_epochs=self.num_epoch, capacity=1000, shuffle=True, seed=0)
            label_queue = tf.train.input_producer(labels, num_epochs=self.num_epoch, capacity=1000, shuffle=True, seed=0)
            
            #filename_queue2 = tf.train.input_producer(filenames, num_epochs=self.num_epoch, capacity=1000, shuffle=True, seed=1)
            #label_queue2 = tf.train.input_producer(labels, num_epochs=self.num_epoch, capacity=1000, shuffle=True, seed=1)
            
        else:
            filenames = self.test_filenames
            labels = self.test_labels
            filename_queue = tf.train.input_producer(filenames, num_epochs=self.num_epoch, capacity=1000, shuffle=True, seed=0)
            label_queue = tf.train.input_producer(labels, num_epochs=self.num_epoch, capacity=1000, shuffle=True, seed=0)
            #filename_fifo = tf.train.string_input_producer(filenames, num_epochs=1, shuffle = True, capacity = len(filenames))
        
        print(len(filenames))
        
        
        
        common_queue = tf.PaddingFIFOQueue(capacity=1000,
                                           dtypes=[tf.float32, tf.int32],
                                           shapes=[(self.config.binary_embed_height, self.config.binary_embed_width, 1), ()])
        
        """
        enqueue_ops = []
        num_readers = 8
        for i in range(num_readers):
            fn = filename_queue.dequeue()
            content = tf.read_file(fn)
            #reader = tf.WholeFileReader()
            #content, filepath = reader.read(filename_queue)
            image = self.preprocessor(content)
            #image = tf.Print(image, [image], summarize=200)
            #label = filepath
            label = label_queue.dequeue()
            enqueue_ops.append(common_queue.enqueue([image, label]))
        qr = tf.train.QueueRunner(common_queue, enqueue_ops)
        #queue_runner.add_queue_runner(queue_runner.QueueRunner(common_queue, enqueue_ops))
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
        image_dequeued, label_dequeued = common_queue.dequeue()
        
        """
        
        fn = filename_queue.dequeue()
        content = tf.read_file(fn)
        image_dequeued = self.preprocessor(content)

        label_dequeued = label_queue.dequeue()
        

        batch_size = self.batch_size
        
        min_after_dequeue = 50
        capacity = min_after_dequeue + 3 * self.batch_size
        image_batch, label_batch = tf.train.shuffle_batch([image_dequeued, label_dequeued],
                                                                batch_size = batch_size, 
                                                                min_after_dequeue=min_after_dequeue, 
                                                                capacity=capacity, 
                                                                num_threads = NUM_PREPROCESS_THREADS)
        
        #label_batch = tf.py_func(self.get_label, [label_batch], [tf.int64])[0]
        #image_batch = tf.concat(axis=0, values=[t_x, w_x])
        #label_batch = tf.concat(axis=0, values=[t_y, w_y])
        
        #label_batch = tf.Print(label_batch, [label_batch], summarize=5000)
        
        return image_batch, label_batch
    
    def prepare_inference(self):
        
        self.filenames = tf.placeholder(tf.string, (None))
        
        self.filepath_queue = tf.FIFOQueue(10000, tf.string, shapes=(None))
        
        self.enqueue = self.filepath_queue.enqueue_many(self.filenames)
        
        
        reader = tf.WholeFileReader()
        
        filename, content = reader.read_up_to(self.filepath_queue, 1)
        
        image = tf.map_fn(self.preprocessor, content, dtype=tf.float32)
        print(image.get_shape())
        
        return image, filename
    
    
        


    def parse_examples(self, serialized_examples):
        '''
        parsed_features = tf.parse_example(serialized_examples, self.features)
        parsed_label_features = tf.parse_example(serialized_examples, self.label_feature)
        print(parsed_features['image_raw'])
        images = tf.map_fn(self.postprocess_for_images, parsed_features['image_raw'].values)
        labels = tf.map_fn(self.postprocess_for_labels, parsed_label_features['label'].values)
        '''
        #print('serialized_exampels', serialized_examples)
        images = tf.map_fn(lambda v: self.process(v)[0], serialized_examples)
        labels = tf.map_fn(lambda v: self.process(v)[1], serialized_examples)
        return images, labels
    
        return self.process(serialized_examples)

    def process(self, serialized_examples):
        print('example', serialized_examples)
        image_raws, labels, shapes = self.parse_example(serialized_examples)
        images = self.postprocess_for_images(image_raws.values)
        labels = self.postprocess_for_labels(labels)
        return images, labels

    def postprocess_for_images(self, image_raws):
        images = self.reshape(image_raws)
        images = self.cast_to_float32(images)
        return images

    def postprocess_for_labels(self, labels):
        if labels != None:
            labels = self.cast_to_int64(labels)
        return labels

    def parse_example(self, serialized_examples):
        parsed_features = tf.parse_single_example(serialized_examples,
                                           self.features)
        image_raws = parsed_features['image_raw']
        #labels = parsed_features['label']
        shapes = dict()
        shapes['height'] = parsed_features['height']
        shapes['width'] = parsed_features['width']
        shapes['depth'] = parsed_features['depth']
        labels = None
        parsed_label_feature = tf.parse_single_example(serialized_examples,
                                                       self.label_feature)
        labels = parsed_label_feature['label']
        return image_raws, labels, shapes

    def decode(self, raw_images):
        images = tf.decode_raw(raw_images, tf.uint8)
        return images

    def reshape(self, images, shapes=None):
        height = 224
        width = 224
        depth= 1
        image_dim = tf.stack([height, width,depth])
        images = tf.reshape(images, image_dim)

        return images

    def cast_to_float32(self, images):
        images = tf.cast(images, tf.float32)
        return images
    def cast_to_int64(self, labels):
        labels = tf.cast(labels, tf.int64)
        return labels