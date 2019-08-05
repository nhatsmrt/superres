from nntoolbox.learner import Learner
from nntoolbox.utils import load_model, get_device
from nntoolbox.callbacks import Callback, CallbackHandler
from nntoolbox.metrics import Metric
from typing import Iterable, Dict
import torch
from torch import Tensor
import torch.nn.functional as F


class SuperResolutionLearner(Learner):
    def learn(
            self, n_epoch: int, upscale_factor: int, callbacks: Iterable[Callback]=None,
            metrics: Dict[str, Metric]=None, final_metric: str='accuracy', load_path=None
    ) -> float:

        assert upscale_factor > 1

        if load_path is not None:
            load_model(self._model, load_path)

        self.scale_factor = 1.0 / upscale_factor

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

    def learn_one_iter(self, high_res: Tensor):
        low_res = F.interpolate(high_res, scale_factor=self.scale_factor)

        data = self._cb_handler.on_batch_begin({'high_res': high_res, 'low_res': low_res}, True)
        high_res, low_res = data['high_res'], data['low_res']
        loss = self._cb_handler.after_losses({"loss": self.compute_loss(low_res, high_res, True)}, True)["loss"]

        if self._cb_handler.on_backward_begin():
            loss.backward()
        if self._cb_handler.after_backward():
            self._optimizer.step()
            if self._cb_handler.after_step():
                self._optimizer.zero_grad()

            if get_device().type == 'cuda':
                mem = torch.cuda.memory_allocated(get_device())
                self._cb_handler.on_batch_end({"loss": loss.cpu(), "allocated_memory": mem})
            else:
                self._cb_handler.on_batch_end({"loss": loss})

    def generate(self, low_res: Tensor, train: bool) -> Tensor:
        return self._cb_handler.after_outputs({"generated": self._model(low_res)}, train)["generated"]

    def compute_loss(self, low_res: Tensor, high_res: Tensor, train: bool) -> Tensor:
        generated = self.generate(low_res, train)
        return self._criterion(generated, high_res)

    @torch.no_grad()
    def evaluate(self):
        self._model.eval()
        tags = []
        imgs = []
        outputs = []
        labels = []

        for high_res in self._val_data:
            low_res = F.interpolate(high_res, scale_factor=(self.scale_factor, self.scale_factor))
            data = self._cb_handler.on_batch_begin({'high_res': high_res, 'low_res': low_res}, False)
            high_res, low_res = data['high_res'], data['low_res']
            generated = self.generate(low_res, False)

            outputs.append(generated)
            labels.append(high_res)

            for i in range(min(len(low_res), 8)):
                imgs.append(high_res[i])
                imgs.append(low_res[i])
                imgs.append(generated[i])
                tags.append("high_res_" + str(i))
                tags.append("low_res_" + str(i))
                tags.append("generated_res_" + str(i))
            break

        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)

        return self._cb_handler.on_epoch_end({"draw": imgs, "tag": tags, "outputs": outputs, "labels": labels})

