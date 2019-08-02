from nntoolbox.learner import Learner
from nntoolbox.utils import load_model
from nntoolbox.callbacks import Callback, CallbackHandler
from nntoolbox.metrics import Metric
from typing import Iterable, Dict
from torch import Tensor
# from torch.nn import Module
# from torch.optim import Optimizer
# from torch.utils.data import DataLoader


class SuperResolutionLearner(Learner):
    def learn(
            self,
            n_epoch: int, callbacks: Iterable[Callback]=None,
            metrics: Dict[str, Metric]=None, final_metric: str='accuracy', load_path=None
    ) -> float:
        if load_path is not None:
            load_model(self._model, load_path)

        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics, final_metric)
        self._cb_handler.on_train_begin()

        for e in range(n_epoch):
            self._model.train()
            self._cb_handler.on_epoch_begin()

            for images in self._train_data:
                self.learn_one_iter(images)

            stop_training = self.evaluate()
            if stop_training:
                print("Patience exceeded. Training finished.")
                break

        return self._cb_handler.on_train_end()

    def learn_one_iter(self, images: Tensor): pass

    def evaluate(self): pass
