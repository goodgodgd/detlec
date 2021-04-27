

class Logger:
    def __init__(self):
        pass

    def save_log(self, epoch, train_log, val_log):
        print("save_log() will be implemented")
        pass


class ModelLog:
    def __init__(self):
        self.frame = dict()
        self.batch = dict()
        self.epoch = dict()

    def append_batch_result(self, outputs):
        pass

    def append_epoch_result(self, **kwargs):
        pass

    def clear(self):
        pass

