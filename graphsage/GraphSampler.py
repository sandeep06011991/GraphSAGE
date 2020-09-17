
class GraphKHopSampler():

    def __init__(self,G, layer_infos):
        self.G = G
        self.layer_infos = layer_infos
        self.support_sizes =  []
        p = 1
        for l in range(len(layer_infos)):
            self.support_sizes.append(p * layer_infos[l].num_samples)
            p = self.support_sizes[l]
        assert(self.support_sizes == [25,250])
    # Scratch function, multiplies the list of nodes to match a khop sampling.
    # Function has to be replaced with next door
    # nodes of batch_size
    # returns k_hop sampled batch_size * hop1 * hop2
    def getKHopSamples(self, nodes):
        khop = {}
        for i in range(len(self.layer_infos)):
            size = 0
            output = []
            target = self.support_sizes[i] * len(nodes)
            while(size < target):
                if(size+ len(nodes) <= target):
                    output.extend(nodes[:])
                    size = size +len(nodes)
                else:
                    output.extend(nodes.copy()[:(target-size)])
                    size = size + target - size
            khop[("hop{}".format(i+1))] = output
        return khop
