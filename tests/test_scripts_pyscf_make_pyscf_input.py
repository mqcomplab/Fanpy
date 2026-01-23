from fanpy.scripts.pyscf.make_pyscf_input import make_pyscf_input

def test_make_pyscf_input(tmp_path):
    # simple H2 test to check if the generated input runs without error
    geom = [['H', ['0.0000000000', '0.0000000000', '0.7500000000']], 
        ['H', ['0.0000000000', '0.0000000000', '0']]]
    basis = "sto6g"
    script_path = str(tmp_path / "script.py")
    file_output = make_pyscf_input(geom, basis)
    with open(script_path, 'w') as f:
        f.write(file_output)
        f.write("\n")
    import subprocess
    subprocess.check_output(["python", script_path])
