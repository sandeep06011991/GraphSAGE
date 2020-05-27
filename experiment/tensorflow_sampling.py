
import numpy as np
import sys,os
from os import path
sys.path.insert(0, os.getcwd())

def use_constant_adj_matrix(dataset,adj_matrix,train_nodes,gpu_avail):
    from graphsage.neigh_samplers import UniformNeighborSampler
    import tensorflow as tf
    a,b = (adj_matrix.shape)
    adj_matrix_constant = tf.constant(adj_matrix)
    adj_matrix_constant = tf.cast(adj_matrix_constant,tf.int32)
    sampler = UniformNeighborSampler(adj_matrix)
    batch =  tf.placeholder(tf.int32, shape=(None,), name='batch')
    sample1 = sampler((batch, 25))
    sample1 = tf.cast(sample1, dtype=tf.int64)
    s_tv = tf.reshape(sample1, [tf.shape(sample1)[0] * 25, ])
    sample2 = sampler((s_tv, 10))
    sess = tf.Session()
    train_nodes =  np.random.permutation(train_nodes)
    import time
    start_time = time.time()
    i = 0
    batchsize = 1
    interrupted = False
    while  i < len(train_nodes):
        offset = min(len(train_nodes),i+batchsize)
        feed_dict= {batch:train_nodes[i:offset]}
        sess.run(sample2, feed_dict)
        i = i+offset
        if time.time()-start_time > 120:
            interrupted = True
            break
    end_time = time.time()
    sess.close()
    with open("tf_measurements.txt","a") as fp:
        print("{} | {} | {} | {} | {} \n".format(dataset,gpu_avail,"CONSTANT",interrupted,end_time-start_time))
        if not interrupted:
            fp.write("{} | {} | {} | {} | {} \n".format(dataset,gpu_avail,"CONSTANT",interrupted,end_time-start_time))
        else:
            fp.write("{} | {} | {} | {} | {} \n".format(dataset,gpu_avail,"CONSTANT",interrupted,(end_time-start_time)*(len(train_nodes)/offset)))


def use_feed_dict_matrix(dataset,adj_matrix,train_nodes,gpu_avail):
    from graphsage.neigh_samplers import UniformNeighborSampler
    import tensorflow as tf
    adj_matrixph = tf.placeholder(tf.int32,shape=adj_matrix.shape,name='adj_matrix')
    sampler = UniformNeighborSampler(adj_matrixph)
    batch =  tf.placeholder(tf.int32, shape=(None,), name='batch')
    sample1 = sampler((batch, 25))
    sample1 = tf.cast(sample1, dtype=tf.int64)
    s_tv = tf.reshape(sample1, [tf.shape(sample1)[0] * 25, ])
    sample2 = sampler((s_tv, 10))
    sess = tf.Session()
    train_nodes =  np.random.permutation(train_nodes)
    import time
    start_time = time.time()
    i = 0
    batchsize = 1
    interrupted = False
    while  i < len(train_nodes):
        offset = min(len(train_nodes),i+batchsize)
        feed_dict= {batch:train_nodes[i:offset],adj_matrixph:adj_matrix}
        sess.run(sample2, feed_dict)
        i = i+offset
        if time.time()-start_time > 120:
            interrupted = True
            break
    end_time = time.time()
    sess.close()
    with open("tf_measurements.txt","a") as fp:
        print("{} | {} | {} | {} | {} \n".format(dataset,gpu_avail,"FEEDDICT",interrupted,(end_time-start_time)*(len(train_nodes)/offset)))
        if not interrupted:
            fp.write("{} | {} | {} | {} | {} \n".format(dataset,gpu_avail,"FEEDDICT",interrupted,end_time-start_time))
        else:
            fp.write("{} | {} | {} | {} | {} \n".format(dataset,gpu_avail,"FEEDDICT",interrupted,(end_time-start_time)*(len(train_nodes)/offset)))    

def run():
    dataset = sys.argv[1]
    gpu = sys.argv[2]

    if gpu == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    fp = open(dataset,'r')
    nodes = {}
    maxv = 0
    for line in fp:
        if line.startswith("#"):
            continue
        a,b = line.split()
        a,b = int(a),int(b)
        if nodes.has_key(a):
            nodes[a].append(b)
        else:
            nodes[a] = [b]
        maxv = max(a,maxv)
        maxv = max(b,maxv)
    max_degree = 128
    print("nodes {} {}".format(maxv,len(nodes)))
    adj_matrix =  np.zeros((maxv+1,max_degree))
    train_nodes = []
    for n in nodes:
        train_nodes.append(n)
        neighbors = np.array(nodes[n])
        if len(neighbors) == 0:
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        adj_matrix[n , :] = neighbors
    a,b = (adj_matrix.shape)
    gpu_constant_possible = False
    if a*b < 2 * 1024 * 1024 * 1024 / 4 :
        gpu_constant_possible = True
    import tensorflow as tf
    gpu_avail = tf.test.is_gpu_available()

    use_feed_dict_matrix(dataset,adj_matrix,train_nodes,gpu_avail)
    if gpu_constant_possible:
        use_constant_adj_matrix(dataset,adj_matrix,train_nodes,gpu_avail)

if __name__ == "__main__":
    run()
