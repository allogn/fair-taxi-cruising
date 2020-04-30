from framework.solvers.Solver import Solver

class DummySolver(Solver):
    '''
    Solver for testing
    '''

    def train(self, db_save_callback = None):
        pass

    def test(self):
        pass

    def load(self):
        pass

    def save(self):
        pass
