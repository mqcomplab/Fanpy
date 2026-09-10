"""Test the new features added to fanpy.eqn.base.BaseSchrodinger.

Covers:
- `wrapped_get_overlaps`: must match `wrapped_get_overlap` called per-determinant (the paper's
  Eq. 9-10 correctness condition), both when the wavefunction has no native batch method (scalar
  fallback) and when it does (native batch path).
- `assign_params`: must skip `component.assign_params` for components whose proposed parameter
  array is unchanged (Eq. 28: "assign only if p_new != p_old").
- `save_params`: non-root MPI ranks must not write parameter files.
- `get_energy_one_proj`: the chunked-refwfn branch must agree with the plain list branch for the
  same total determinant set (Eq. 30-31), and results must be invariant to how work is
  distributed and reduced across MPI ranks (Eq. 32-33), both with and without chunking.

HDF5-backed projection spaces are intentionally NOT covered here (feature still under active
development).

A note on the MPI tests: real multi-rank testing normally requires `mpi4py` installed and running
under `mpiexec -n N`, which doesn't fit a single-process `pytest` CI job. Instead,
`ThreadedFakeComm` below emulates a communicator using real Python threads and a barrier, so that
`allgather` genuinely blocks until all "ranks" contribute -- exercising the actual reduction logic
in `get_energy_one_proj`, not just the `mpi_comm is None` short-circuit. This is a reasonable
substitute for correctness testing, but it is not a replacement for running the real test suite
under real MPI at least once (e.g. in a separate, opt-in CI job) before relying on it in
production.
"""
import os
import threading

from fanpy.eqn.base import BaseSchrodinger
from fanpy.eqn.utils import ParamContainer
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np

import pytest

from utils import disable_abstract


def _make_wfn_ham():
    """Build a small wfn/ham pair shared by several tests."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    return wfn, ham


# ---------------------------------------------------------------------------
# wrapped_get_overlaps
# ---------------------------------------------------------------------------


def test_baseschrodinger_wrapped_get_overlaps_matches_scalar():
    """wrapped_get_overlaps (scalar-fallback path) must match wrapped_get_overlap per sd.

    This is the paper's minimal validation condition (Eq. 9-10): the batched overlap array and
    Jacobian must equal what the scalar interface produces one determinant at a time.
    """
    wfn, ham = _make_wfn_ham()
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ParamContainer(3), [True])]
    )
    sds = np.array([0b0101, 0b0110, 0b1100])

    scalar_vals = np.array([test.wrapped_get_overlap(sd) for sd in sds])
    batch_vals = test.wrapped_get_overlaps(sds)
    assert np.allclose(batch_vals, scalar_vals)

    scalar_jac = np.array([test.wrapped_get_overlap(sd, deriv=True) for sd in sds])
    batch_jac = test.wrapped_get_overlaps(sds, deriv=True)
    assert np.allclose(batch_jac, scalar_jac)

    with pytest.raises(TypeError):
        test.wrapped_get_overlaps(sds, deriv=1)


def test_baseschrodinger_wrapped_get_overlaps_uses_native_batch_method():
    """When the wavefunction exposes `get_overlaps`, wrapped_get_overlaps must call it (rather
    than looping `get_overlap`), and the result must still match the scalar path.
    """
    wfn, ham = _make_wfn_ham()
    test = disable_abstract(BaseSchrodinger)(wfn, ham)
    sds = np.array([0b0101, 0b0110, 0b1100])

    calls = []

    def fake_get_overlaps(sds_arg, deriv=None):
        calls.append(deriv)
        if deriv is None:
            return np.array([wfn.get_overlap(sd) for sd in sds_arg])
        return np.array([wfn.get_overlap(sd, deriv) for sd in sds_arg])

    wfn.get_overlaps = fake_get_overlaps

    result = test.wrapped_get_overlaps(sds)
    assert len(calls) == 1
    assert np.allclose(result, [wfn.get_overlap(sd) for sd in sds])

    result_deriv = test.wrapped_get_overlaps(sds, deriv=True)
    assert len(calls) == 2
    scalar_jac = np.array([test.wrapped_get_overlap(sd, deriv=True) for sd in sds])
    assert np.allclose(result_deriv, scalar_jac)


# ---------------------------------------------------------------------------
# assign_params cache preservation
# ---------------------------------------------------------------------------


def test_baseschrodinger_assign_params_skips_unchanged_components():
    """assign_params must not call component.assign_params when its proposed parameter array is
    unchanged from the current one (Eq. 28: "assign only if p_new != p_old").
    """
    wfn, ham = _make_wfn_ham()
    param1 = ParamContainer(np.array([1.0, 2.0]))
    param2 = ParamContainer(np.array([3.0, 4.0]))
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(param1, np.array([0, 1])), (param2, np.array([0, 1]))]
    )

    calls = {"param1": 0, "param2": 0}
    orig1, orig2 = param1.assign_params, param2.assign_params

    def spy1(p):
        calls["param1"] += 1
        orig1(p)

    def spy2(p):
        calls["param2"] += 1
        orig2(p)

    param1.assign_params = spy1
    param2.assign_params = spy2

    # Same values as current params -> neither component should be reassigned.
    test.assign_params(np.array([1.0, 2.0, 3.0, 4.0]))
    assert calls == {"param1": 0, "param2": 0}

    # Change only param2's values -> only param2 should be reassigned.
    test.assign_params(np.array([1.0, 2.0, 30.0, 4.0]))
    assert calls == {"param1": 0, "param2": 1}
    assert np.allclose(param2.params, [30.0, 4.0])

    # Change only param1's values -> only param1 should be reassigned.
    test.assign_params(np.array([10.0, 2.0, 30.0, 4.0]))
    assert calls == {"param1": 1, "param2": 1}
    assert np.allclose(param1.params, [10.0, 2.0])


# ---------------------------------------------------------------------------
# save_params MPI gating
# ---------------------------------------------------------------------------


class FakeComm:
    """Minimal stand-in for an mpi4py communicator; only Get_rank is needed by save_params."""

    def __init__(self, rank):
        self._rank = rank

    def Get_rank(self):
        """Return the fake rank."""
        return self._rank


def test_baseschrodinger_save_params_mpi_nonroot_skips(tmp_path):
    """Non-root MPI ranks must not write parameter files; the root rank still does."""
    wfn, ham = _make_wfn_ham()
    tmpfile = str(tmp_path / "temp.npy")

    test = disable_abstract(BaseSchrodinger)(wfn, ham, tmpfile=tmpfile, mpi_comm=FakeComm(1))
    test.save_params()
    assert not os.path.isfile(str(tmp_path / "temp_CIWavefunction.npy"))

    test.mpi_comm = FakeComm(0)
    test.save_params()
    assert os.path.isfile(str(tmp_path / "temp_CIWavefunction.npy"))


# ---------------------------------------------------------------------------
# get_energy_one_proj: chunking correctness (no MPI needed)
# ---------------------------------------------------------------------------


class FakeChunkedSpace:
    """Minimal stand-in for a chunked projection space (e.g. a future HDF5SlaterSpace)."""

    def __init__(self, chunks):
        self._chunks = chunks

    def iter_chunks(self):
        """Yield stored chunks."""
        yield from self._chunks


def test_baseschrodinger_get_energy_one_proj_chunked_matches_unchunked():
    """The chunked-refwfn branch must give the same energy/gradient as passing the full
    determinant list directly (both implement the same projected-reference form, Eq. 14/17), since
    chunking is only a partition of the same sum (Eq. 30-31).
    """
    wfn, ham = _make_wfn_ham()
    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]

    test = disable_abstract(BaseSchrodinger)(wfn, ham)
    flat_energy = test.get_energy_one_proj(sds)
    flat_grad = test.get_energy_one_proj(sds, deriv=True)

    # Chunks of uneven size, including one empty chunk, to exercise the empty-chunk skip.
    chunked_space = FakeChunkedSpace([sds[:2], [], sds[2:4], sds[4:]])
    chunked_energy = test.get_energy_one_proj(chunked_space)
    chunked_grad = test.get_energy_one_proj(chunked_space, deriv=True)

    assert np.allclose(chunked_energy, flat_energy)
    assert np.allclose(chunked_grad, flat_grad)


# ---------------------------------------------------------------------------
# get_energy_one_proj: MPI reduction correctness
# ---------------------------------------------------------------------------


class ThreadedFakeComm:
    """A fake MPI communicator using real Python threads so that `allgather` (a genuine blocking
    collective) can be exercised without requiring real mpi4py / `mpiexec`.

    One instance is created per simulated rank; all instances for a given run share the same
    barrier/buffer/lock so that `allgather` calls made by different "ranks" (threads) actually
    rendezvous, mirroring what a real MPI allgather does.
    """

    def __init__(self, rank, size, barrier, buffer, lock):
        self.rank = rank
        self.size = size
        self._barrier = barrier
        self._buffer = buffer
        self._lock = lock
        self._call_index = 0

    def Get_rank(self):
        """Return this simulated rank."""
        return self.rank

    def Get_size(self):
        """Return the total number of simulated ranks."""
        return self.size

    def allgather(self, payload):
        """Block until every simulated rank has contributed, then return all payloads in rank order."""
        call_index = self._call_index
        self._call_index += 1
        with self._lock:
            self._buffer.setdefault(call_index, {})[self.rank] = payload
        self._barrier.wait(timeout=30)
        gathered = [self._buffer[call_index][r] for r in range(self.size)]
        self._barrier.wait(timeout=30)
        return gathered


def _run_ranks(size, target):
    """Run `target(rank, comm)` concurrently across `size` fake MPI ranks (real threads).

    Returns the list of each rank's return value, ordered by rank. Re-raises the first exception
    encountered on any rank (after unblocking the others via barrier.abort()).
    """
    barrier = threading.Barrier(size)
    buffer = {}
    lock = threading.Lock()
    results = [None] * size
    errors = [None] * size

    def worker(rank):
        comm = ThreadedFakeComm(rank, size, barrier, buffer, lock)
        try:
            results[rank] = target(rank, comm)
        except threading.BrokenBarrierError:
            # This rank was unblocked because another rank hit a real error and called
            # barrier.abort(); that other rank's exception (captured below) is the real
            # cause, so don't let this downstream artifact overwrite/mask it.
            pass
        except BaseException as err:  # noqa: BLE001 - surfaced via errors list below
            errors[rank] = err
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for err in errors:
        if err is not None:
            raise err
    return results


def test_baseschrodinger_get_energy_one_proj_mpi_matches_serial():
    """The MPI-reduced one-sided projected energy (and gradient) must match the serial result for
    the same reference/projection space (Eq. 32-33): MPI only changes how the same sum is
    distributed and reduced across ranks, not the total.
    """
    wfn, ham = _make_wfn_ham()
    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001]

    serial = disable_abstract(BaseSchrodinger)(wfn, ham)
    serial_energy = serial.get_energy_one_proj(sds)
    serial_grad = serial.get_energy_one_proj(sds, deriv=True)

    def run_on_rank(rank, comm):
        test = disable_abstract(BaseSchrodinger)(wfn, ham, mpi_comm=comm)
        energy = test.get_energy_one_proj(sds)
        grad = test.get_energy_one_proj(sds, deriv=True)
        return energy, grad

    results = _run_ranks(3, run_on_rank)

    for energy, grad in results:
        assert np.allclose(energy, serial_energy)
        assert np.allclose(grad, serial_grad)


def test_baseschrodinger_get_energy_one_proj_chunked_mpi_matches_serial():
    """Chunked refwfn combined with MPI (the combination highlighted in the paper, Sec. IX) must
    give the same energy/gradient as the chunked-but-serial result.
    """
    wfn, ham = _make_wfn_ham()
    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001]
    chunks = [sds[:2], sds[2:4], sds[4:]]

    serial = disable_abstract(BaseSchrodinger)(wfn, ham)
    serial_energy = serial.get_energy_one_proj(FakeChunkedSpace(chunks))
    serial_grad = serial.get_energy_one_proj(FakeChunkedSpace(chunks), deriv=True)

    def run_on_rank(rank, comm):
        test = disable_abstract(BaseSchrodinger)(wfn, ham, mpi_comm=comm)
        energy = test.get_energy_one_proj(FakeChunkedSpace(chunks))
        grad = test.get_energy_one_proj(FakeChunkedSpace(chunks), deriv=True)
        return energy, grad

    results = _run_ranks(2, run_on_rank)

    for energy, grad in results:
        assert np.allclose(energy, serial_energy)
        assert np.allclose(grad, serial_grad)

