import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from ase import Atoms
from ase.io.trajectory import Trajectory

from torched_tacaw import core


class NestedConfig:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, keys):
        current = self.data
        if isinstance(keys, tuple):
            for key in keys:
                current = current[key]
            return current
        return current[keys]

    @staticmethod
    def crop_arr_qspace_2ROI(array, from_shape="full"):
        return array


class FakeTrajectory:
    def __init__(self, atoms, length=1):
        self.atoms = atoms
        self.length = length

    def __getitem__(self, index):
        return self.atoms.copy()

    def __len__(self):
        return self.length


def make_atoms():
    return Atoms(
        "Si",
        positions=[[1.0, 1.0, 1.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def close_readers(calculator):
    for reader in getattr(calculator, "trajectories", []):
        reader.trajectory.close()


def make_calculator(number_of_beads):
    calculator = core.Calculator.__new__(core.Calculator)
    calculator.logger = core.tools.NullLogger()
    calculator.device = "cpu"
    calculator.config = NestedConfig(
        {
            "trajectory": {
                "chunks": {
                    "size": 1,
                    "step": 1,
                    "starts": [0],
                },
            },
            "simulation": {
                "center_atoms_in_cell": False,
                "n_slices": 1,
                "kspace": {
                    "shape_full": [2, 2],
                    "bandwidth_limiting": [2 / 3, 2 / 3],
                },
            },
            "beam": {"energy_keV": 60.0},
        }
    )
    calculator.batch_params = {"trajectory_chunk_id": 0}
    calculator.trajectories = [
        FakeTrajectory(make_atoms())
        for _ in range(number_of_beads)
    ]
    calculator.trajectory = calculator.trajectories[0]
    # Two probe positions ensure the coherent average preserves STEM scan axes.
    calculator.init_waves_flat = np.zeros((2, 2, 2), dtype=np.complex128)
    calculator.final_wavefunctions = torch.empty(
        (1, 1, 2, 2, 2),
        dtype=torch.complex128,
    )
    return calculator


def config_kwargs():
    return {
        "datafolder": "/tmp/torched-tacaw-test/",
        "beam_energy_keV": 60.0,
        "beam_conv_ang_mrad": 20.0,
        "scanning_mode": "parallelogram",
        "scanning_shape": [1, 1],
        "scanning_batch_shape": [1, 1],
        "scanning_origin": [0.0, 0.0],
        "scanning_basis_vectors": [[1.0, 0.0], [0.0, 1.0]],
        "sample_name": "Si",
        "sample_temperature_K": 300.0,
        "sample_structure_file": "structure.extxyz",
        "trajectory_timestep_fs": 1.0,
        "trajectory_chunks_size": 2,
        "trajectory_chunks_skip_init": 0,
        "trajectory_chunks_step": 1,
        "trajectory_chunks_nof": 1,
        "trajectory_chunks_overlap": 1.0,
        "kspace_shape_full": [4, 4],
        "kspace_ROI_mode": "center",
        "kspace_ROI_shape": [2, 2],
        "kspace_bandwidth_limiting": [2 / 3, 2 / 3],
        "n_slices": 1,
        "frequency_THz_ROI": [-500.0, 500.0],
    }


class MultiBeadCalculatorTests(unittest.TestCase):
    def test_load_trajectory_accepts_files_only_yaml_config(self):
        config = core.Config(hardload=True, trajectory={"files": ["bead0.traj", "bead1.traj"]})
        with TemporaryDirectory() as folder:
            path = str(Path(folder) / "config.yaml")
            config.dump_to_yaml(path)
            calculator = core.Calculator.__new__(core.Calculator)
            calculator.config = core.Config.load_from_yaml(path)
            calculator.logger = core.tools.NullLogger()
            with patch.object(
                core.io, "TrajctoryReader",
                side_effect=[FakeTrajectory(make_atoms()), FakeTrajectory(make_atoms())],
            ):
                calculator.load_trajectory()
            self.assertEqual(calculator.nof_beads, 2)

    def test_multislice_uses_chunk_start_and_step_for_every_bead(self):
        with TemporaryDirectory() as folder:
            paths = [str(Path(folder) / f"bead{i}.traj") for i in range(2)]
            for bead, path in enumerate(paths):
                with Trajectory(path, "w") as trajectory:
                    for frame in range(5):
                        atoms = make_atoms()
                        atoms.positions[0, 0] = frame + 2 * bead
                        trajectory.write(atoms)
            calculator = make_calculator(2)
            self.addCleanup(close_readers, calculator)
            calculator.config.data["trajectory"] = {
                "file": paths[0], "files": paths,
                "chunks": {"size": 2, "step": 2, "starts": [1]},
            }
            calculator.final_wavefunctions = torch.empty(
                (1, 2, 2, 2, 2), dtype=torch.complex128,
            )
            calculator.load_trajectory()

            # Replace expensive scattering with a position-dependent complex wave.
            # At frames 1 and 3 the bead-average x positions are 2 and 4 angstrom.
            def structure(cell, atomlist, *args):
                return atomlist[0, 0] * cell[0]

            def precursor(crystal, *args, **kwargs):
                return crystal, None

            def multislice(waves, slices, position, transmission, **kwargs):
                return torch.full((2, 2, 2), complex(position, 1), dtype=torch.complex128)

            with (
                patch.object(core.pyms, "structure", side_effect=structure),
                patch.object(core.pyms, "multislice_precursor", side_effect=precursor),
                patch.object(core.pyms, "multislice", side_effect=multislice),
            ):
                calculator.perform_multislice()
            torch.testing.assert_close(
                calculator.final_wavefunctions[0, :, 0, 0, 0],
                torch.tensor([2 + 1j, 4 + 1j], dtype=torch.complex128),
            )

    def test_config_rejects_empty_bead_list(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            core.Config(**config_kwargs(), trajectory_files=[])

    def test_config_rejects_conflicting_single_and_multi_file_inputs(self):
        with self.assertRaisesRegex(ValueError, "must equal"):
            core.Config(
                **config_kwargs(), trajectory_file="different.traj",
                trajectory_files=["bead0.traj", "bead1.traj"],
            )

    def test_config_accepts_single_path_and_offline_length(self):
        with TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.traj"
            with patch.object(core.ase.io, "read", return_value=make_atoms()):
                config = core.Config(
                    **config_kwargs(), trajectory_files=missing, trajectory_len=8,
                )
            self.assertEqual(config["trajectory", "files"], [str(missing)])
            self.assertEqual(config["trajectory", "chunks", "starts"], [0])

    def test_load_real_beads_rejects_mismatched_atoms_and_cells(self):
        for mismatch in ("atoms", "cell", "length"):
            with self.subTest(mismatch=mismatch), TemporaryDirectory() as folder:
                paths = [str(Path(folder) / f"bead{i}.traj") for i in range(2)]
                atoms = make_atoms()
                other = atoms.copy()
                if mismatch == "atoms":
                    other.set_atomic_numbers([6])
                elif mismatch == "cell":
                    other.set_cell([11.0, 10.0, 10.0])
                for path, frame in zip(paths, (atoms, other)):
                    with Trajectory(path, "w") as trajectory:
                        trajectory.write(frame)
                        if mismatch == "length" and path == paths[1]:
                            trajectory.write(frame)
                calculator = core.Calculator.__new__(core.Calculator)
                self.addCleanup(close_readers, calculator)
                calculator.logger = core.tools.NullLogger()
                calculator.config = NestedConfig(
                    {"trajectory": {"file": paths[0], "files": paths}}
                )
                with self.assertRaises(ValueError):
                    calculator.load_trajectory()

    def test_old_single_file_config_loads_real_trajectory(self):
        with TemporaryDirectory() as folder:
            path = str(Path(folder) / "single.traj")
            with Trajectory(path, "w") as trajectory:
                trajectory.write(make_atoms())
            calculator = core.Calculator.__new__(core.Calculator)
            self.addCleanup(close_readers, calculator)
            calculator.logger = core.tools.NullLogger()
            calculator.config = NestedConfig({"trajectory": {"file": path}})
            calculator.load_trajectory()
            np.testing.assert_array_equal(
                calculator.trajectory[0].positions, [[1.0, 1.0, 1.0]],
            )
            self.assertEqual(calculator.nof_beads, 1)

    def test_config_records_ordered_bead_files(self):
        kwargs = config_kwargs()
        kwargs["trajectory_files"] = ["bead0.traj", "bead1.traj"]
        fake_trajectories = [
            FakeTrajectory(make_atoms(), length=8),
            FakeTrajectory(make_atoms(), length=8),
        ]

        with (
            patch.object(
                core.io,
                "TrajctoryReader",
                side_effect=fake_trajectories,
            ),
            patch.object(core.ase.io, "read", return_value=make_atoms()),
        ):
            config = core.Config(**kwargs)

        self.assertEqual(
            config["trajectory", "files"],
            ["bead0.traj", "bead1.traj"],
        )
        self.assertEqual(config["trajectory", "file"], "bead0.traj")
        self.assertEqual(config["trajectory", "nof_beads"], 2)

    def test_config_rejects_unequal_bead_lengths(self):
        kwargs = config_kwargs()
        kwargs["trajectory_files"] = ["bead0.traj", "bead1.traj"]

        with patch.object(
            core.io,
            "TrajctoryReader",
            side_effect=[
                FakeTrajectory(make_atoms(), length=8),
                FakeTrajectory(make_atoms(), length=7),
            ],
        ):
            with self.assertRaisesRegex(
                ValueError,
                "same number of frames",
            ):
                core.Config(**kwargs)

    def test_multislice_coherently_averages_beads_and_preserves_scan_axis(self):
        calculator = make_calculator(number_of_beads=2)
        bead_0 = torch.stack(
            [
                torch.full((2, 2), 1.0, dtype=torch.complex128),
                torch.full((2, 2), 3.0, dtype=torch.complex128),
            ]
        )
        bead_1 = torch.stack(
            [
                torch.full((2, 2), -1.0, dtype=torch.complex128),
                torch.full((2, 2), 1.0, dtype=torch.complex128),
            ]
        )

        with (
            patch.object(core.pyms, "structure", return_value=object()),
            patch.object(
                core.pyms,
                "multislice_precursor",
                return_value=(object(), object()),
            ),
            patch.object(
                core.pyms,
                "multislice",
                side_effect=[bead_0, bead_1],
            ),
        ):
            calculator.perform_multislice()

        expected = torch.stack(
            [
                torch.zeros((2, 2), dtype=torch.complex128),
                torch.full((2, 2), 2.0, dtype=torch.complex128),
            ]
        )
        torch.testing.assert_close(
            calculator.final_wavefunctions[0, 0],
            expected,
        )

    def test_single_bead_path_preserves_exit_wave(self):
        calculator = make_calculator(number_of_beads=1)
        exit_wave = torch.stack(
            [
                torch.full((2, 2), 2.0 + 1.0j, dtype=torch.complex128),
                torch.full((2, 2), 4.0 - 1.0j, dtype=torch.complex128),
            ]
        )

        with (
            patch.object(core.pyms, "structure", return_value=object()),
            patch.object(
                core.pyms,
                "multislice_precursor",
                return_value=(object(), object()),
            ),
            patch.object(core.pyms, "multislice", return_value=exit_wave),
        ):
            calculator.perform_multislice()

        torch.testing.assert_close(
            calculator.final_wavefunctions[0, 0],
            exit_wave,
        )

    def test_load_trajectory_accepts_old_single_file_config(self):
        calculator = core.Calculator.__new__(core.Calculator)
        calculator.logger = core.tools.NullLogger()
        calculator.config = NestedConfig(
            {"trajectory": {"file": "single.traj"}}
        )
        fake_trajectory = FakeTrajectory(make_atoms(), length=4)

        with patch.object(
            core.io,
            "TrajctoryReader",
            return_value=fake_trajectory,
        ) as reader:
            calculator.load_trajectory()

        reader.assert_called_once_with("single.traj", None)
        self.assertEqual(calculator.nof_beads, 1)
        self.assertIs(calculator.trajectory, calculator.trajectories[0])


if __name__ == "__main__":
    unittest.main()
