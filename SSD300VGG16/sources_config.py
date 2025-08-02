
from typing import Tuple, List

class SourceConfig:
    def __init__(self, tail_size: Tuple[int, int], dboxNum: int, channels: int, aspects: List[float]):
        self._tail_size = tail_size
        self._dboxNum = dboxNum
        self._channels = channels
        self._aspects = aspects
    
    @property
    def tail_size(self):
        return self._tail_size
    
    @property
    def dboxNum(self):
        return self._dboxNum
    
    @property
    def channels(self):
        return self._channels
    
    @property
    def aspects(self):
        return self._aspects

source1_config = SourceConfig((38, 38), 4,  512, [1, 2, 0.5])
source2_config = SourceConfig((19, 19), 6, 1024, [1, 2, 3, 0.5, 1/3])
source3_config = SourceConfig((10, 10), 6,  512, [1, 2, 3, 0.5, 1/3])
source4_config = SourceConfig((5, 5),   6,  256, [1, 2, 3, 0.5, 1/3])
source5_config = SourceConfig((3, 3),   4,  256, [1, 2, 0.5])
source6_config = SourceConfig((1, 1),   4,  256, [1, 2, 0.5])
sources_config = [source1_config, 
                  source2_config, 
                  source3_config, 
                  source4_config, 
                  source5_config, 
                  source6_config]