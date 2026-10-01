"""Regression tests for low-temperature energy-gain spectra and FFT bin alignment."""
import os
import unittest
import numpy as np
import scipy.constants as constants
import scipy.signal
import torch
from torched_tacaw import core


def impulse_spectrum(size=600, roi=None, temperature=10.0):
    """A window-compensated impulse has unit spectral power at every FFT bin."""
    start, stop = roi or (0, size)
    config = core.Config(
        hardload=True,
        trajectory={"chunks": {"size": size}, "timestep_effective_fs": 7.5},
        simulation={
            "window": "hann",
            "frequency_THz": {"full": [-1000 / 15, 1000 / 15],
                              "ROI_indices": [start, stop]},
            "kspace": {"shape_full": [1, 1]},
        },
        sample={"temperature_K": temperature},
        beam={"scanning": {"batch_shape": [1, 1]}},
    )
    calculator = core.Calculator.__new__(core.Calculator)
    calculator.config = config
    calculator.logger = core.tools.NullLogger()
    calculator.device = os.environ.get("TACAW_TEST_DEVICE", "cpu")
    wave = torch.zeros((1, size, 1, 1, 1), dtype=torch.complex128,
                       device=calculator.device)
    wave[0, size // 2, 0, 0, 0] = 1 / scipy.signal.windows.hann(size)[size // 2]
    calculator.final_wavefunctions = wave
    calculator.perform_tacaw()
    return calculator.tacaw.cpu().numpy().ravel() / (2 * np.pi / size**2)


class ThermalPrefactorTests(unittest.TestCase):
    def test_10k_energy_gain_tail_does_not_drop_to_zero(self):
        spectrum = impulse_spectrum()
        # All 600 bins are representable in float64 at 10 K (down to exp(-320)).
        self.assertTrue(np.all(np.isfinite(spectrum)))
        self.assertTrue(np.all(spectrum > 0), "spurious zero in 10 K spectrum")

    def test_zero_energy_bin_has_unit_thermal_factor(self):
        self.assertAlmostEqual(impulse_spectrum()[300], 1.0, places=13)

    def test_matches_double_precision_reference_on_actual_fft_bins(self):
        for size, roi in [(600, None), (599, None), (600, (200, 340))]:
            with self.subTest(size=size, roi=roi):
                energies_joule = np.fft.fftshift(np.fft.fftfreq(size, 7.5e-15)) * constants.h
                x = energies_joule / (constants.k * 10.0)
                expected = np.ones_like(x)
                nonzero = x != 0
                expected[nonzero] = x[nonzero] / (-np.expm1(-x[nonzero]))
                if roi:
                    expected = expected[slice(*roi)]
                # atol=0 is essential: an absolute tolerance hides the tiny tail.
                np.testing.assert_allclose(impulse_spectrum(size, roi), expected,
                                           rtol=2e-12, atol=0)


if __name__ == "__main__":
    unittest.main()
