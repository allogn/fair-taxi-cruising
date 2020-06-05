import tensorflow as tf
import numpy as np
from stable_baselines.common.callbacks import BaseCallback
from collections.abc import Iterable
import logging

class EpisodeStatsLogger:
    def __init__(self, tb_writer):
        self.writer = tb_writer

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
    
    def write(self, stats, step):
        values = []
        # if len(stats['rewards']) == 0:
        #     return True
        for k, val in stats.items():
            if k == 'rewards':
                assert len(val) > 0
                values.append(tf.Summary.Value(tag='reward/' + k, histo=self.create_hist(val)))
                values.append(tf.Summary.Value(tag='reward/' + k + '_mean', simple_value=float(np.mean(val))))
                values.append(tf.Summary.Value(tag='reward/' + k + '_std', simple_value=float(np.std(val))))
                values.append(tf.Summary.Value(tag='reward/' + k + '_sum', simple_value=float(np.sum(val))))
                continue
            if isinstance(val, Iterable):
                assert len(val) > 0, k
                values.append(tf.Summary.Value(tag='stats/' + k, histo=self.create_hist(val)))
                values.append(tf.Summary.Value(tag="stats/" + k + '_mean', simple_value=float(np.mean(val))))
                values.append(tf.Summary.Value(tag="stats/" + k + '_min', simple_value=float(np.min(val))))
                values.append(tf.Summary.Value(tag="stats/" + k + '_sum', simple_value=float(np.sum(val))))
            else:
                assert type(val) == float
                assert type(k) == str
                values.append(tf.Summary.Value(tag="stats/" + k, simple_value=val))
        summary = tf.Summary(value=values)
        self.writer.add_summary(summary, step)
        self.writer.flush()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)
        self.episodes_recorded = set()

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        Might not be ther end of an episode

        Retrieve epoch data from environment and plot it
        """
        w = EpisodeStatsLogger(self.locals['writer'])
        stats = self.model.get_env().env_method("get_episode_info")[-1] 
        # -1 because env is vectorized, and we take result only from the last env in the vector (can be any)
        if stats == None:
            # on first rollouts it might happen that none of the environments have finished any of episodes
            # but we don't allow this
            raise Exception("n_steps is too small, first rollout completed without termination")
        
        # self.rollout_calls would be a wrong iteration id here, because there might be 
        # many rollout calls before an episode is finished in one of the environments
        episode_id = self.model.get_env().env_method("get_resets")[-1]
        # assert that solver is updated by a new episode (although several might have passed
        assert episode_id not in self.episodes_recorded
        w.write(stats, len(self.episodes_recorded))
        self.episodes_recorded.add(episode_id)
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

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.solver.run_tests(0, draw=self.draw, verbose=self.verbose)

    def _on_rollout_end(self) -> bool:
        if self.eval_freq > 0 and self.rollout_calls % self.eval_freq == 0:
            self.solver.run_tests(self.rollout_calls // self.eval_freq, draw=self.draw, verbose=self.verbose)
        return True

class RobustCallback(BaseCallback):
    def __init__(self, solver, nu, epsilon, gamma, cmin, cmax, verbose=0):
        super(RobustCallback, self).__init__(verbose)
        self.solver = solver
        self.nu = nu
        self.epsilon = epsilon
        self.gamma = gamma
        self.cmin = cmin
        self.cmax = cmax
        self.call = 0

    def find_c(self):
        cmin = self.cmin
        cmax = self.cmax
        c = (cmax + cmin) / 2
        steps_log = []
        while abs((cmax + cmin)/2 - cmin) > self.epsilon:
            self.solver.test_env.set_income_bound(c)
            stats = self.solver.run_test_episode(0, draw=False, debug=False) 
            reward = np.sum(stats['driver_income_bounded'])

            robust_threshold = c * self.solver.test_env.n_drivers * (1 - self.nu)
            possible = reward > robust_threshold
            steps_log.append((c, reward, c * self.solver.test_env.n_drivers, reward - robust_threshold))
            if possible:
                cmin = cmin + (c - cmin) * self.gamma
                c = (cmax + cmin) / 2
            else:
                cmax = cmax - (cmax - c) * self.gamma
                c = (cmax + cmin) / 2

        logging.info("Finishing with final c={}".format(c))
        steps_log = sorted(steps_log)
        self.solver.log['step_log_{}'.format(self.rollout_calls)] = np.array(steps_log, dtype=float).tolist()

        return c

    def _on_rollout_end(self) -> bool:
        if self.rollout_calls % 3 == 0:
            c = self.find_c()
            self.training_env.env_method("set_income_bound", c)
            # self.solver.test_env.set_income_bound(c)