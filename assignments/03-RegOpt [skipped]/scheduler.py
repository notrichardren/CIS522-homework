from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        num_epochs,
        initial_learning_rate,
        final_learning_rate,
        last_epoch=-1,
    ):
        """
        Create a new scheduler. Note to students: You can change the arguments to this constructor, if you need to add new parameters.

        final_learning_rate must be greater than initial_learning_rate

        """
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        super(CustomLRScheduler, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)
        # Here's our dumb baseline implementation: return [i for i in self.base_lrs]

        if self.last_epoch <= 0:
            return [
                group["lr"] * float(self.initial_learning_rate)
                for group in self.optimizer.param_groups
            ]
        else:
            current_learning_rate = float(self.initial_learning_rate) - (
                float(self.last_epoch)
                * (self.initial_learning_rate - self.final_learning_rate)
                / float(self.num_epochs)
            )
            return [
                group["lr"] * current_learning_rate
                for group in self.optimizer.param_groups
            ]
