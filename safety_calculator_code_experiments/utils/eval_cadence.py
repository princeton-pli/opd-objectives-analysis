from transformers import TrainerCallback


class FractionalEpochEvalCallback(TrainerCallback):
    """Evaluate on the first optimizer step that crosses each 1/N epoch boundary."""

    def __init__(self, evals_per_epoch: int) -> None:
        if evals_per_epoch < 1:
            raise ValueError("evals_per_epoch must be positive.")
        self.evals_per_epoch = evals_per_epoch
        self._last_boundary = 0

    def on_train_begin(self, args, state, control, **kwargs):
        del args, kwargs
        epoch = 0.0 if state.epoch is None else float(state.epoch)
        self._last_boundary = int((epoch + 1e-12) * self.evals_per_epoch)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.epoch is None:
            return control
        boundary = int((float(state.epoch) + 1e-12) * self.evals_per_epoch)
        if boundary > self._last_boundary:
            control.should_evaluate = True
            self._last_boundary = boundary
        return control
