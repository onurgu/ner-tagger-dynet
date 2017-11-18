import os

class DynetSaver():

    def __init__(self, parameter_collection, checkpoint_dir):
        self.parameter_collection = parameter_collection
        self.checkpoint_dir = checkpoint_dir

    def save(self, epoch=None, n_bests=None):
        assert epoch or (n_bests >= 0), "One of epoch or n_bests should be specified"
        model_dir_path = "model-epoch-%08d" % epoch if epoch is not None else ("best-models-%08d" % n_bests)
        model_checkpoint_dir_path = os.path.join(self.checkpoint_dir, model_dir_path)
        if not os.path.exists(model_checkpoint_dir_path):
            os.mkdir(model_checkpoint_dir_path)
        self.parameter_collection.save(os.path.join(model_checkpoint_dir_path,
                                                    "model.ckpt"))

    def get_newest_ckpt_directory(self):
        model_directories = sorted(os.walk(self.checkpoint_dir).next()[1])

        if model_directories:
            return os.path.join(self.checkpoint_dir, model_directories[-1])
        else:
            return None

    def restore(self, ckpt_directory):
        self.parameter_collection.populate(os.path.join(ckpt_directory, "model.ckpt"))
