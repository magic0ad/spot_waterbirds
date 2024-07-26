# Author: Alexandre Drouin
# Adapted by: Sebastien Lachapelle

import json
import numpy as np
import os
import queue

from time import strftime, time


try:
    import comet_ml
    COMET_AVAIL = True
except:
    COMET_AVAIL = False

try:
    import wandb
    WANDB_AVAIL = True
except:
    WANDB_AVAIL = False

try:
    import tensorboardX
    TBX_AVAIL = True
except:
    TBX_AVAIL = False


def _check_randomstate(random_state):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    if not isinstance(random_state, np.random.RandomState):
        raise ValueError("Random state must be numpy RandomState or int.")
    return random_state


def _prefix_stage(metrics, stage):
    if stage is not None:
        metrics = {"%s/%s" % (stage, k): v for k, v in metrics.items()}
    return metrics


class WandBLogger(object):
    def __init__(self, experiment):
        self.experiment = experiment

    def log_figure(self, stage, step, name, figure):
        prefix = ""
        #if stage is not None:
        #    prefix += stage + "/"
        #if step is not None:
        #    prefix += str(step) + "/"
        self.experiment.log({prefix + name: figure, "step": step}, step=step)

    def log_metrics(self, stage, step, metrics):
        metrics["step"] = step
        self.experiment.log(metrics, step=step)


class CometLogger(object):
    def __init__(self, experiment):
        self.experiment = experiment

    def log_figure(self, stage, step, name, figure):
        prefix = ""
        #if stage is not None:
        #    prefix += stage + "/"
        #if step is not None:
        #    prefix += str(step) + "/"
        try:
            self.experiment.log_figure(figure_name=prefix + name, figure=figure, step=step, overwrite=False)
        except:
            self.experiment.log_image(figure, name=prefix + name, step=step, overwrite=False)

    def log_metrics(self, stage, step, metrics):
        self.experiment.log_metrics(step=step, dic=_prefix_stage(metrics, stage))


class JsonLogger(object):
    def __init__(self, path, time=True, max_fig_save=None):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.time = time
        self.max_fig_save = max_fig_save  # makes sure we don't save too many .png files.
        self.current_figs = {}

    def log_metrics(self, stage, step, metrics):
        metrics["stage"] = stage
        metrics["step"] = step
        if self.time:
            metrics["time"] = time()
        with open(os.path.join(self.path, "log.ndjson"), "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def log_figure(self, stage, step, name, figure):
        # if stage is not None:
        #    prefix += stage + "/"
        # if step is not None:
        #    prefix += str(step) + "/"
        if name not in self.current_figs.keys():
            self.current_figs[name] = queue.Queue()

        file_name = f"{name}_{step}.png"
        figure.savefig(os.path.join(self.path, file_name))
        self.current_figs[name].put(file_name)

        # removing old figures
        if self.current_figs[name].qsize() > self.max_fig_save:
            file_to_delete = self.current_figs[name].get()
            os.remove(os.path.join(self.path, file_to_delete))


class StdoutLogger(object):
    def __init__(self, time=True):
        self.time = time

    def log_metrics(self, stage, step, metrics):
        prefix = "" if stage is None else (stage + " -- ")
        try:
            log = f"{prefix}{step}:\t" + "\t".join("%s: %.6f" % (m, v) for m, v in metrics.items())
        except:
            log = f"{prefix}{step}:\t" + "\t".join("%s: %s" % (m, v) for m, v in metrics.items())
        if self.time:
            log += "\t time: " + strftime('%X %x')
        print(log)


class TensorboardXLogger(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    def log_metrics(self, stage, step, metrics):
        for m, v in _prefix_stage(metrics, stage).items():
            self.summary_writer.add_scalar(m, v, step)


# XXX: I'll eventually move this class to the shared code-base
class UniversalLogger(object):
    """
    A logger that simultaneously logs to multiple outputs.

    Parameters:
    -----------
    comet: comet_ml.Experiment
        A comet experiment
    json: str
        The path to a json file
    stdout: bool
        Whether or not to print metrics to the standard output
    tensorboardx: tensorboardx.SummaryWriter
        A summary writer for tensorboardx
    time: bool
        Whether to log the time in some loggers (supported: json, stdout)
    throttle: int
        The minimum time between logs in seconds

    """
    def __init__(self, wandb=None, comet=None, json=None, stdout=False, tensorboardx=None, time=True, throttle=None, max_fig_save=None):
        super().__init__()
        loggers = []
        if wandb is not None:
            if not WANDB_AVAIL:
                raise RuntimeError("wandb is not available on this platform. Please install it.")
            loggers.append(WandBLogger(experiment=wandb))
        if comet is not None:
            if not COMET_AVAIL:
                raise RuntimeError("comet_ml is not available on this platform. Please install it.")
            loggers.append(CometLogger(experiment=comet))
        if json is not None:
            loggers.append(JsonLogger(json, time=time, max_fig_save=max_fig_save))
        if stdout:
            loggers.append(StdoutLogger(time=time))
        if tensorboardx is not None:
            if not TBX_AVAIL:
                raise RuntimeError("TensorboardX is not available on this platform. Please install it.")
            loggers.append(TensorboardXLogger(tensorboardx))
        assert len(loggers) >= 1
        self.loggers = loggers

        # Throttling
        self.throttle = throttle
        self.last_log_time = 0
        self.current_stage = None

    def _check_stage(self, stage):
        # This will force logging if we are entering a new stage
        if self.current_stage != stage:
            self.last_log_time = 0
            self.current_stage = stage

    def _check_throttle(self):
        return self.throttle is None or time() - self.last_log_time > self.throttle

    def log_metrics(self, step, metrics, stage=None, throttle=True):
        """
        Log a real-valued metric

        Parameters:
        -----------
        stage: str
            The current stage of execution (e.g., "train", "test", etc.)
        step: uint
            The current step number (e.g., epoch)
        metrics: dict
            A dictionnary with metric names as keys and metric values as values
        throttle: bool
            Whether to respect log throttling or not (default is True)

        """
        self._check_stage(stage)
        if self._check_throttle() or not throttle:
            for log in self.loggers:
                if hasattr(log, "log_metrics"):
                    log.log_metrics(stage, step, dict(metrics))
            self.last_log_time = time()

    def log_figure(self, name, figure, stage=None, step=None, throttle=True):
        """
        Log a matplotlib figure

        Parameters:
        -----------
        name: str
            The name of the figure
        figure: mpl.Figure
            The matplotlib figure
        step: uint
            The current step number (default: None). If required by a logger and not provided, an exception will be
            raised.
        throttle: bool
            Whether to respect log throttling or not (default is True)

        """
        self._check_stage(stage)
        if self._check_throttle() or not throttle:
            for log in self.loggers:
                if hasattr(log, "log_figure"):
                    log.log_figure(stage, step, name, figure)
            self.last_log_time = time()