from fanpy.scripts.gaussian.run_calc import run_calc
from utils import find_datafile
import subprocess

def test_run_calc():
    oneint = find_datafile("data/data_h2_hf_sto6g_oneint.npy")
    twoint = find_datafile("data/data_h2_hf_sto6g_twoint.npy")
    # attempt to run simple calculation with minimal inputs
    # this checks if the default parameters are working correctly
    run_calc(
        nelec=2,
        one_int_file=oneint,
        two_int_file=twoint,
        wfn_type="cisd")

def test_run_calc_cli():
    oneint = find_datafile("data/data_h2_hf_sto6g_oneint.npy")
    twoint = find_datafile("data/data_h2_hf_sto6g_twoint.npy")
    subprocess.check_output([
        "fanpy_run_calc",
        "--nelec", "2",
        "--one_int_file", oneint,
        "--two_int_file", twoint,
        "--wfn_type", "cisd",
    ])