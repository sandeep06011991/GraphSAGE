import sys,os
from os import path
sys.path.insert(0, os.getcwd())
import time
from graphsage.utils import load_data, run_random_walks
from graphsage.minibatch import NodeMinibatchIterator,EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
import tensorflow as tf

PREFIX = "./example_data/toy-ppi"
file = None
N_WALKS = 50

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('gpu', 0, "which gpu to use.")
# tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                             """Whether to log device placement.""")
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

results = {}

def add_to_dict(k,v):
    global results
    results[k] = v

def create_measurement_file():
    e = path.exists("experiments.txt")
    global results
    global file
    file = open("experiments.txt", 'a')
    if not e:
        file.write ("Dataset | GPU | Sup-Epoch | Nextdoor-epoch \n")# print header
        pass
    # print values from dictionary
    file.write("{} | {} | {}| {} \n".format(results['DATASET'],results['GPU'],
                                     results['SEPOCH'],results['NEXTSEPOCH']))
    file.close()


def supervised_epoch_time(PREFIX):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    from graphsage.supervised_train import train
    FLAGS.train_prefix = PREFIX
    FLAGS.model = 'graphsage_mean'
    FLAGS.sigmoid = True
    train_data = load_data(FLAGS.train_prefix)
    time = train(train_data)
    add_to_dict('SEPOCH',time)

def nextdoor_supervised_epoch_time(PREFIX):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    del_all_flags(tf.flags.FLAGS)
    from graphsage.sampled_supervised_train import  train
    FLAGS.train_prefix = PREFIX
    FLAGS.model = 'graphsage_mean'
    FLAGS.sigmoid = True
    train_data = load_data(FLAGS.train_prefix)
    time = train(train_data)
    print("NEXTSEPOCH {}".format(time))
    add_to_dict('NEXTSEPOCH', time)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def run():
    global PREFIX
    import sys
    PREFIX = sys.argv[1]
    add_to_dict("DATASET",(PREFIX))
    is_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    add_to_dict ("GPU",is_available)
    supervised_epoch_time(PREFIX)
    nextdoor_supervised_epoch_time(PREFIX)
    create_measurement_file()
    print("All Done !!! ")


'''
    how to run !!!
    python experiment/epoch_run_time.py ./example_data/toy-ppi
'''
if __name__ == "__main__":
    # print("hello world")
    run()
    # run_single_experiment()
