import numpy as np

from typing import List, Union


class CostMonitor:
    
    def __init__(self):
        self._cost_values = {}
        
    def log_cost(self,
                 name: str,
                 values: Union[np.ndarray, List[float]]):
        if isinstance(values, list):
            values = np.array(values)
        else:
            values = values.copy()
        if name in self._cost_values:
            self._cost_values[name].append(values)
        else:
            self._cost_values[name] = [values]
    
    def get(self,
            name: str) -> np.ndarray:
        return np.stack(self._cost_values[name], axis=0)
