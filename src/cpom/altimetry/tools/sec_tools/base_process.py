
class BaseProcessor: 
    def __init__(self, data):
        self.data = data

    def process(self):
        raise NotImplementedError("Subclasses should implement this method")