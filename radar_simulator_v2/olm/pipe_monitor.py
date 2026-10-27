import os
import select
import struct
import numpy as np
import matplotlib.pyplot as plt

from .signal_processing import dpc, cfar_1d, range_doppler_power


class PipeMonitor:
    def __init__(self, monitor, pipe_path: str = "/tmp/radar_pipe"):
        self.monitor = monitor
        self.pipe_path = pipe_path
        self.buffer = bytearray()

        # Header is two 32-bit unsigned integers (npri, nrg) -> 8 bytes total
        # '<II' means Little-Endian, two unsigned ints
        self.header_format = "<II"
        self.header_size = struct.calcsize(self.header_format)

        # State tracking
        self.reading_payload = False
        self.current_npri = 0
        self.current_nrg = 0
        self.expected_payload_size = 0

        self._setup_pipe()

    def _setup_pipe(self):
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)
        self.pipe_fd = os.open(self.pipe_path, os.O_RDONLY | os.O_NONBLOCK)
        print(f"[PipeMonitor] Listening on {self.pipe_path}. Ready for dynamic dimensions...")

    def run(self):
        try:
            while True:
                r, _, _ = select.select([self.pipe_fd], [], [], 0.01)

                if self.pipe_fd in r:
                    chunk = os.read(self.pipe_fd, 1024 * 1024)
                    if not chunk:
                        plt.pause(0.1)
                        continue
                    self.buffer.extend(chunk)

                self._process_buffer()
                plt.pause(0.01)

        except KeyboardInterrupt:
            print("[PipeMonitor] Shutting down.")
            self.monitor.close()

    def _process_buffer(self):
        """Processes the buffer state machine (Header -> Payload)."""
        while True:
            if not self.reading_payload:
                # STATE 1: Waiting for Header
                if len(self.buffer) >= self.header_size:
                    header_bytes = self.buffer[:self.header_size]
                    del self.buffer[:self.header_size]

                    self.current_npri, self.current_nrg = struct.unpack(self.header_format, header_bytes)

                    # Dynamically calculate payload size for this specific frame
                    # 8 bytes for your custom float-based complex_t (4 bytes real + 4 bytes imag)
                    self.sz_raw = self.current_npri * self.current_nrg * 8
                    # self.sz_dpc = self.current_nrg * 4  # 4 bytes for float
                    # self.sz_fft = self.current_npri * self.current_nrg * 8
                    # self.sz_cfar = self.current_nrg * 4  # 4 bytes for int
                    self.sz_stats = 24  # 3 doubles (8 bytes each)

                    # self.expected_payload_size = self.sz_raw + self.sz_dpc + self.sz_fft + self.sz_cfar + self.sz_stats
                    self.expected_payload_size = self.sz_raw + self.sz_stats
                    self.reading_payload = True
                else:
                    break  # Wait for more bytes to complete the header

            if self.reading_payload:
                # STATE 2: Waiting for Payload
                if len(self.buffer) >= self.expected_payload_size:
                    payload_bytes = self.buffer[:self.expected_payload_size]
                    del self.buffer[:self.expected_payload_size]

                    self._unpack_and_update(payload_bytes)

                    # Reset state to look for the next frame's header
                    self.reading_payload = False
                else:
                    break  # Wait for more bytes to complete the payload

    def _unpack_and_update(self, raw_bytes: memoryview | bytes):
        offset = 0
        npri = self.current_npri
        nrg = self.current_nrg

        raw_sig_bytes = raw_bytes[offset: offset + self.sz_raw]
        raw_signal = np.frombuffer(raw_sig_bytes, dtype=np.complex64).reshape(npri, nrg)
        offset += self.sz_raw

        # dpc_bytes = raw_bytes[offset: offset + self.sz_dpc]
        # dpc_avg = np.frombuffer(dpc_bytes, dtype=np.float64)
        # offset += self.sz_dpc
        #
        # fft_bytes = raw_bytes[offset: offset + self.sz_fft]
        # fft_signal = np.frombuffer(fft_bytes, dtype=np.complex64).reshape(npri, nrg)
        # offset += self.sz_fft
        #
        # cfar_bytes = raw_bytes[offset: offset + self.sz_cfar]
        # cfar_mask = np.frombuffer(cfar_bytes, dtype=np.float32)
        # offset += self.sz_cfar

        dpc_power = dpc(raw_signal, fs=1e6, pulse_width=1e-6)
        dpc_avg = np.mean(np.abs(dpc_power) ** 2, axis=1)

        fft_signal = range_doppler_power(raw_signal)

        cfar_mask, _ = cfar_1d(dpc_avg, 4, 10, 2.0)

        stat_bytes = raw_bytes[offset: offset + self.sz_stats]
        throughput, elapsed, running_avg = struct.unpack("<ddd", stat_bytes)

        submode_stat = {
            "submode_name": f"Dynamic_{npri}x{nrg}",
            "amount_bytes": self.expected_payload_size,
            "throughput_gbps": throughput,
            "elapsed_ms": elapsed,
            "running_avg_gbps": running_avg,
        }

        self.monitor.update(raw_signal, dpc_avg, fft_signal, cfar_mask, submode_stat)