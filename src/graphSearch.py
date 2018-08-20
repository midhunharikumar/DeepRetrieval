import hnswlib


class GraphSearch():

    def __init__(self,):
        self.index = hnswlib.Index(space='l2', dim=4096)

    def create_index(self, items, labels):
        self.index.init_index(max_elements=items.shape[
                              0], ef_construction=200, M=16)
        self.index.add_items(items, labels)
        self.index.set_ef(50)

    def save_index(self):
        self.index.save_index("deep_index.bin")

    def load_index(self, index_filename='deep_index.bin'):
        self.index.load_index(index_filename)

    def knn(self, query):
        nbrs = self.index.knn_query(query, k=1)
        return nbrs
