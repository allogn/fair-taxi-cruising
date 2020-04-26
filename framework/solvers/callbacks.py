import tensorflow as tf
import numpy as np
from stable_baselines.common.callbacks import BaseCallback
from collections.abc import Iterable

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        pass

    def create_hist(self, values):
        counts, bin_edges = np.histogram(values)#, bins=30)
        values = np.array(values)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        
        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        return hist

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.

        Retrieve epoch data from environment and plot it
        """
        values = []
        stats_vec = self.model.get_env().env_method("get_episode_info")
        stats = stats_vec[0]
        if len(stats['rewards']) == 0:
            return True
        for k, val in stats.items():
            if k == 'rewards':
                values.append(tf.Summary.Value(tag='reward/' + k, histo=self.create_hist(val)))
                values.append(tf.Summary.Value(tag='reward/' + k + '_mean', simple_value=float(np.mean(val))))
                values.append(tf.Summary.Value(tag='reward/' + k + '_std', simple_value=float(np.std(val))))
                values.append(tf.Summary.Value(tag='reward/' + k + '_sum', simple_value=float(np.sum(val))))
                continue
            if isinstance(val, Iterable):
                values.append(tf.Summary.Value(tag='stats/' + k, histo=self.create_hist(val)))
                values.append(tf.Summary.Value(tag="stats/" + k + '_mean', simple_value=float(np.mean(val))))
                values.append(tf.Summary.Value(tag="stats/" + k + '_std', simple_value=float(np.std(val))))
                values.append(tf.Summary.Value(tag="stats/" + k + '_sum', simple_value=float(np.sum(val))))
            else:
                assert type(val) == float
                assert type(k) == str
                values.append(tf.Summary.Value(tag="stats/" + k, simple_value=val))
        summary = tf.Summary(value=values)
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True  


class TestingCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, solver, verbose=0, eval_freq=1, draw=True):
        super(TestingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.solver = solver
        self.draw = draw
        self.verbose = verbose

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.solver.run_tests(self.n_calls, draw=self.draw, verbose=self.verbose)