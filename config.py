import os 
    
class Config(object):

    def __init__(self, checkpoint_dir="/root/data/dict/refactoring/checkpoint",
                 binary_embed_width=32,
                 binary_embed_height=12000,
                 feature_maps=[100,200,300,400,500],
                 kernels=[1,2,3,4,5],
                 batch_size=16,
                 classes_num=25,
                 dropout_prob=0.5,
                 forward_only=False,
                 dataset_name="raw",
                 data_dir="/root/data/dict/PE",
                 epoch=10,
                 learning_rate=0.0001,
                 decay=0.5,
                 preprocessor="thumbnail",
                 specialist=False,
                 whitelist=False,
                 data_type="tfrecord",
                 summary_iter=100,
                 save_iter=1000,
                 label_query="",
                 data_query=""):
    
        if not os.path.exists(checkpoint_dir):
            print(" [*] Creating checkpoint directory...")
            os.makedirs(checkpoint_dir)
            
            
        self.checkpoint_dir = checkpoint_dir
        self.binary_embed_width = binary_embed_width
        self.binary_embed_height = binary_embed_height
        self.feature_maps = feature_maps
        self.kernels = kernels
        self.batch_size = batch_size
        self.num_classes = classes_num
        self.dropout_prob = dropout_prob
        self.forward_only = forward_only
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.decay = decay
        self.preprocessor=preprocessor
        self.specialist=specialist
        self.whitelist=whitelist
        self.data_type=data_type
        self.summary_iter=summary_iter
        self.save_iter=save_iter
        self.label_query=label_query
        self.data_query=data_query