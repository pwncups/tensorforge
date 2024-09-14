from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def get_layer_info(self):
        pass

    @abstractmethod
    def load_metadata(self):
        pass