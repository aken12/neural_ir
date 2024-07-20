from torch.utils.data import IterableDataset

class IndexDataset(IterableDataset):
    def __init__(self, collection_path):
        super().__init__()
        self.collection_path = collection_path

    def __iter__(self):
        with open(self.collection_path, "r") as f:
            for line in f:
                line = line.strip().split('\t') # doc_id, text
                if len(line) == 1:
                    line.append("")
                yield line