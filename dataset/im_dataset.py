from utils import DiskIODataset


class ImTrainingSet(DiskIODataset):
    def __init__(self, opts):
        opts['if_train'] = True
        opts['if_baseline'] = False
        super().__init__(**opts)

class ImTestSet(DiskIODataset):
    def __init__(self, opts):
        opts['if_train'] = False
        opts['if_baseline'] = False
        super().__init__(**opts)

class ImTrainingSet_base(DiskIODataset):
    def __init__(self, opts):
        opts['if_train'] = True
        opts['if_baseline'] = True
        super().__init__(**opts)

class ImTestSet_base(DiskIODataset):
    def __init__(self, opts):
        opts['if_train'] = False
        opts['if_baseline'] = True
        super().__init__(**opts)
