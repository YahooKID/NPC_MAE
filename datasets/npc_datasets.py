from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NPC_Dataset(BaseSegDataset):

    METAINFO = dict(
        classes=('T1C', 'T2'),
        palette=[[128, 0, 0], [0, 0, 128]])

    def __init__(self, aeg1, arg2):
        pass