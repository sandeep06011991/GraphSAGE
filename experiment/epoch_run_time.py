import sys,os
sys.path.insert(0, os.getcwd())
import time
from graphsage.utils import load_data, run_random_walks
from graphsage.minibatch import NodeMinibatchIterator,EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
import tensorflow as tf

PREFIX = "./example_data/toy-ppi"
file = None
N_WALKS = 50

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)


def open_log_file():
    global file
    file = open("experiments.txt",'a')
    file.write("Dataset {} ##################### \n".format(PREFIX))

def log(str):
    global file
    print(str)
    file.write(str + "\n")

def close_file():
    global file
    if file is not None:
        file.close()

# # Measure Look up time
def time_to_do_random_walks(G):
    nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
    start_time = time.time()
    run_random_walks(G, nodes, num_walks=N_WALKS)
    end_time = time.time()
    log("CPU random walk time {}".format(end_time - start_time))

def time_for_negative_sampling(minibatch):
    label = tf.cast(minibatch.placeholders["batch"], dtype=tf.int64)
    labels = tf.reshape(label,[tf.shape(label)[0],1])
    neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels,
        num_true=1,
        num_sampled=20,
        unique=False,
        range_max=len(minibatch.deg),
        distortion=0.75,
        unigrams=minibatch.deg.tolist()))
    minibatch.shuffle()
    sess = tf.Session()
    start_time = time.time()
    while not minibatch.end():
        feed_dict, _ = minibatch.next_minibatch_feed_dict()
        sess.run(neg_samples, feed_dict)
    end_time = time.time()
    sess.close()
    log("Negative sampling time {}".format(end_time - start_time))



def getMiniBatchIterator(G, id_map, class_map, num_classes):
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None,), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    minibatch = NodeMinibatchIterator(G,
                                      id_map,
                                      placeholders,
                                      class_map,
                                      num_classes,
                                      batch_size=512,
                                      max_degree=128)
    return minibatch

def time_to_do_adj_matrix_construction(minibatch):
    start_time = time.time()
    minibatch.construct_adj()
    end_time = time.time()
    log("Time to construct adj matrix does not involve gpu{}".format(end_time - start_time))

def time_2_hop_neighbourhood_sampling(minibatch):
    pruned_adj_matrix = tf.constant(minibatch.adj, dtype=tf.int32)
    sampler = UniformNeighborSampler(pruned_adj_matrix)
    sample1 = sampler((minibatch.placeholders["batch"], 10))
    s_tv = tf.reshape(sample1, [tf.shape(sample1)[0] * 10,])
    sample2 = sampler((s_tv,28))
    sess = tf.Session()
    minibatch.shuffle()
    start_time = time.time()
    while not minibatch.end():
        feed_dict,_ = minibatch.next_minibatch_feed_dict()
        sess.run(sample2 , feed_dict)
    end_time = time.time()
    sess.close()
    log("Time to neighbourhood sample 1 and 2 hops {}".format(end_time - start_time))

def run():
    global PREFIX
    import sys
    PREFIX = sys.argv[1]
    print("PREFIX {}".format(PREFIX))
    is_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    open_log_file()
    log("GPU availability {}".format(is_available))
    G, feats, id_map, walks, class_map = load_data(PREFIX)
    print("Number of nodes {}".format(G.number_of_nodes()))
    print("Number of Edges {}".format(G.number_of_edges()))
    import sys
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    time_to_do_random_walks(G)
    minibatch = getMiniBatchIterator(G, id_map, class_map, num_classes)
    time_to_do_adj_matrix_construction(minibatch)
    time_2_hop_neighbourhood_sampling(minibatch)
    time_for_negative_sampling(minibatch)
    close_file()
    print("All Done !!! ")

def print_adj_matrix_to_file():
    global PREFIX
    import sys
    PREFIX = sys.argv[1]
    print("PREFIX {}".format(PREFIX))
    G, feats, id_map, walks, class_map = load_data(PREFIX)
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    minibatch = getMiniBatchIterator(G, id_map, class_map, num_classes)
    adj = minibatch.adj
    fp = open("_adj_matrix","w")
    fp.write("DATASET" + PREFIX + "\n")
    fp.write("nd1 : < list of 128 sampled neighbours> ")
    for ndid in G.nodes():
        fp.write("{} :".format(ndid))
        neighbours = adj[id_map[ndid]]
        for n in neighbours:
            fp.write("{} ".format(n))
        fp.write("\n")
    fp.close()


def run_single_experiment():
    print_adj_matrix_to_file()

'''
    how to run !!!
    python experiment/epoch_run_time.py ./example_data/toy-ppi
'''
if __name__ == "__main__":
    # run()
    run_single_experiment()