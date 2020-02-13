import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'scripts'))

from load_and_run_experiment import *

run_experiment("dummy")
run_experiment("tiny_grid")
