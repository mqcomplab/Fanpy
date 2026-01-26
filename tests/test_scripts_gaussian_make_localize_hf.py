from fanpy.scripts.gaussian.make_localize_hf import make_script
import subprocess

def test_make_script(tmp_path):
    geom = """
    H -1.0 0.0 0.0\n
    H 1.0 0.0 0.0\n
    Be 0.0 1.0 0.0\n
    """
    xyz_file = tmp_path/"molecule.xyz"
    with open(xyz_file, "w") as f:
        f.write(geom)
    
    basis_file = "sto-6g"
    method = None
    system_inds = [1, 1, 2]
    calculate_filename = tmp_path/"calculate.py"
    make_script(xyz_file, basis_file, method, system_inds, filename=calculate_filename)
    result = subprocess.run(["python", calculate_filename])
    assert result.returncode == 0