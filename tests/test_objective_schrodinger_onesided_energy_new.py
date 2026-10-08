"""Test the new features added to fanpy.eqn.energy_oneside.EnergyOneSideProjection.

Covers:
- Chunked projection-space support in `assign_refwfn` (objects exposing `iter_chunks`).
- MPI-aware printing/queuing in `objective` and `gradient` (root vs. non-root rank).
- The `normalize` default change in `gradient` (True -> False, now matching `objective`).
- The `save_params` gating change in `gradient` (now respects `self.step_save`, matching
  `objective`).

HDF5-backed projection spaces are intentionally NOT covered here; that feature is still under
active development (see New_Fanpy_implementations.pdf, "Important notice 2").

True multi-rank MPI behavior (the numerator/denominator reductions across ranks) is also NOT
covered here, since that requires a real `mpi4py` communicator and multi-process execution
(`mpiexec -n N pytest ...`). These tests instead use a minimal fake communicator to exercise the
root/non-root branching logic in `objective`/`gradient`, which does not require `mpi4py` to be
installed and so is safe to run in the existing single-process CI job.
"""
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np

from utils import skip_init


class FakeChunkedSpace:
    """Minimal stand-in for a chunked projection space (e.g. a future HDF5SlaterSpace).

    Only `iter_chunks` is required for `EnergyOneSideProjection.assign_refwfn` to take the
    chunked-space branch; the actual chunk contents are not exercised by these tests.
    """

    def __init__(self, chunks):
        self._chunks = chunks

    def iter_chunks(self):
        """Yield stored chunks."""
        yield from self._chunks


class FakeComm:
    """Minimal stand-in for an mpi4py communicator.

    `Get_rank` is needed for the root/non-root print gating in `objective`/`gradient`.
    `Get_size` is needed because `objective`/`gradient` call into `get_energy_one_proj`
    (in base.py), which checks `comm.Get_size() > 1` to decide whether to actually take the
    MPI-splitting code path. Reporting size=1 here keeps these tests scoped to the print/queue
    gating behavior in energy_oneside.py -- real multi-rank reduction correctness is covered
    separately, by ThreadedFakeComm in test_objective_schrodinger_base_new_features.py.
    """

    def __init__(self, rank, size=1):
        self._rank = rank
        self._size = size

    def Get_rank(self):
        """Return the fake rank."""
        return self._rank

    def Get_size(self):
        """Return the fake communicator size."""
        return self._size


def _make_test(mpi_comm=None, step_print=True, step_save=False):
    """Construct a small EnergyOneSideProjection instance for testing."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    return EnergyOneSideProjection(
        wfn, ham, mpi_comm=mpi_comm, step_print=step_print, step_save=step_save
    )


def test_energy_oneside_assign_refwfn_chunked():
    """Test that a chunked projection-space object bypasses SD/CI validation."""
    test = skip_init(EnergyOneSideProjection)
    test.wfn = CIWavefunction(2, 4)

    chunked_space = FakeChunkedSpace([(0b0101, 0b0110)])
    test.assign_refwfn(refwfn=chunked_space)
    assert test.refwfn is chunked_space


def test_energy_oneside_objective_mpi_root_prints(capsys):
    """Root rank (or no mpi_comm) should print the energy when step_print=True."""
    guess = np.random.rand(6)

    test = _make_test(mpi_comm=None, step_print=True)
    test.objective(guess)
    captured = capsys.readouterr()
    assert "Electronic energy" in captured.out

    test = _make_test(mpi_comm=FakeComm(0), step_print=True)
    test.objective(guess)
    captured = capsys.readouterr()
    assert "Electronic energy" in captured.out


def test_energy_oneside_objective_mpi_nonroot_silent(capsys):
    """Non-root ranks should not print, and should not populate print_queue."""
    guess = np.random.rand(6)

    test = _make_test(mpi_comm=FakeComm(1), step_print=True)
    test.objective(guess)
    captured = capsys.readouterr()
    assert captured.out == ""

    test = _make_test(mpi_comm=FakeComm(1), step_print=False)
    test.objective(guess)
    assert "Electronic energy" not in test.print_queue


def test_energy_oneside_objective_mpi_root_queues():
    """Root rank with step_print=False should still populate print_queue."""
    test = _make_test(mpi_comm=None, step_print=False)
    guess = np.random.rand(6)
    test.objective(guess)
    assert "Electronic energy" in test.print_queue

    test = _make_test(mpi_comm=FakeComm(0), step_print=False)
    test.objective(guess)
    assert "Electronic energy" in test.print_queue


def test_energy_oneside_gradient_mpi_root_prints(capsys):
    """Root rank (or no mpi_comm) should print the gradient norm when step_print=True."""
    guess = np.random.rand(6)

    test = _make_test(mpi_comm=None, step_print=True)
    test.gradient(guess)
    captured = capsys.readouterr()
    assert "Norm of the gradient" in captured.out

    test = _make_test(mpi_comm=FakeComm(0), step_print=True)
    test.gradient(guess)
    captured = capsys.readouterr()
    assert "Norm of the gradient" in captured.out


def test_energy_oneside_gradient_mpi_nonroot_silent(capsys):
    """Non-root ranks should not print, and should not populate print_queue, for gradient()."""
    guess = np.random.rand(6)

    test = _make_test(mpi_comm=FakeComm(1), step_print=True)
    test.gradient(guess)
    captured = capsys.readouterr()
    assert captured.out == ""

    test = _make_test(mpi_comm=FakeComm(1), step_print=False)
    test.gradient(guess)
    assert "Norm of the gradient" not in test.print_queue


def test_energy_oneside_gradient_normalize_default_false():
    """gradient() should default to normalize=False (changed from True).

    This makes the default consistent with objective(), which has always defaulted to
    normalize=False.
    """
    test = _make_test()
    wfn = test.wfn

    calls = []
    wfn.normalize = lambda *args, **kwargs: calls.append((args, kwargs))

    guess = np.random.rand(6)
    test.gradient(guess)
    assert calls == []

    test.gradient(guess, normalize=True)
    assert len(calls) == 1


def test_energy_oneside_gradient_save_respects_step_save():
    """gradient() should only call save_params when save=True AND self.step_save is True.

    Previously, gradient() saved whenever save=True regardless of step_save; this brings it in
    line with objective(), which has always checked both.
    """
    test = _make_test(step_save=False)

    calls = []
    test.save_params = lambda: calls.append(True)

    guess = np.random.rand(6)

    # save=True but step_save=False -> should NOT save.
    test.gradient(guess, save=True)
    assert calls == []

    # save=True and step_save=True -> should save.
    test.step_save = True
    test.gradient(guess, save=True)
    assert calls == [True]

    # save=False -> should never save, regardless of step_save.
    calls.clear()
    test.gradient(guess, save=False)
    assert calls == []

