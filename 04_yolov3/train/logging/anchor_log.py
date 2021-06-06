

class AnchorLog:
    def __init__(self, ckpt_path, epoch):
        self.ckpt_path = ckpt_path
        self.epoch = epoch

    def __call__(self, step, grtr, pred, loss_by_type):
        pass

    def get_result(self):
        pass
