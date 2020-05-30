import unittest
from framework.Experiment import *

class TestGenerator(unittest.TestCase):

    def setUp(self):
        pass

    def testMultiParamGeneration(self):
        params = {
            "tag": "test",
            "data": {
                "dataset_type": "linear",
                "n": [3, [4]]
            },
            "solver": {
                "Diff": [{"draw": 0}]
            },
            "seed": 123,
            "DEBUG": 1,
            "full_rerun": 1
        }

        e = Experiment(params)
        e.clear()
        e.run()

        # one solution, two datasets => 2 solutions in total
        assert e.db.solution.count_documents({"tag": "test"}) == 2
        e.clear()