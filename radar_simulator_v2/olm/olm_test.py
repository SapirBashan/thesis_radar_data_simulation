import time
import numpy as np
from online_monitor import OnlineMonitor

def generate_synthetic_radar_frame(npri: int, nrg: int, step: int) -> np.ndarray:
    """
    Generates a synthetic raw radar frame (npri x nrg) with complex noise
    and a moving target to test the Range-Doppler and DPC plots.
    """
    # 1. Generate complex Gaussian noise floor
    noise_power = 1.0
    noise = np.sqrt(noise_power / 2) * (
            np.random.randn(npri, nrg) + 1j * np.random.randn(npri, nrg)
    )

    # 2. Inject a simulated target
    target_range_bin = 128
    target_snr_linear = 50.0  # Strong target so it pops out on the plots

    # Simulate Doppler shift (target moving across slow-time)
    # The phase changes slightly every frame to simulate continuous motion
    normalized_doppler_freq = 0.15
    pulse_indices = np.arange(npri)

    # Target signal: Phase rotation over slow-time
    target_signal = np.sqrt(target_snr_linear) * np.exp(
        1j * 2 * np.pi * normalized_doppler_freq * pulse_indices + (step * 0.1)
    )

    # Add target to the specific range bin
    raw_signal = noise.copy()
    raw_signal[:, target_range_bin] += target_signal

    # Inject a secondary, weaker, static clutter target at a different range
    clutter_bin = 40
    clutter_signal = np.sqrt(20.0) * np.exp(1j * 2 * np.pi * 0.0 * pulse_indices)  # 0 Doppler
    raw_signal[:, clutter_bin] += clutter_signal

    return raw_signal


def main():
    # Matrix dimensions
    npri = 64  # Number of pulses (slow-time)
    nrg = 256  # Number of range gates (fast-time)

    print("Starting Online Monitor test...")
    print("Press Ctrl+C in the terminal to exit.")

    monitor = OnlineMonitor(show_every=1)

    try:
        step = 0
        while True:
            # Generate mock data
            raw_signal = generate_synthetic_radar_frame(npri, nrg, np.random.randint(0, 100))

            # Mock submode statistics to test the metadata text
            mock_stats = {
                "submode_name": "TEST_TRACK",
                "amount_bytes": raw_signal.nbytes,
                "throughput_gbps": 1.25 + np.random.uniform(-0.1, 0.1),
                "elapsed_ms": 15.4 + np.random.uniform(-1.0, 1.0),
                "running_avg_gbps": 1.24
            }

            # Push to monitor
            monitor.update(raw_signal, submode_stat=mock_stats, step=step)

            # Throttle the loop to simulate a ~20 FPS frame rate
            time.sleep(0.1)
            step += 1

    except KeyboardInterrupt:
        print("\nTest interrupted by user. Closing monitor.")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()