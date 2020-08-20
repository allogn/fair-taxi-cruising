import unittest
from framework.helpers import *

class TestHelpers(unittest.TestCase):

    def test_is_solver_correct(self):
        s1 = "Gym_CouNei0_IncIncToObs1_MinRew0_NInt90_NorRew0_PooFir1_Wc00000_WeiPoo0"
        assert is_solver_correct(s1, "Gym", [("minimum_reward",0)])
        assert not is_solver_correct(s1, "Gym", [("normalize_reward", 1)])