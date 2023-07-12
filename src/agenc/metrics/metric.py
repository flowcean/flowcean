class Metric:
    @property
    def name(self):
        return self.__class__.__name__

    def __call__(self, y, y_, *args):
        raise NotImplementedError
