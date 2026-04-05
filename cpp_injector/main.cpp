//
// Created by Daniel Attali on 05/04/2026.
//
#include <iostream>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <cerrno>

// Ensure strict byte alignment
#pragma pack(push, 1)
struct FrameHeader {
    uint32_t npri;
    uint32_t nrg;
};

struct FrameStats {
    double throughput_gbps;
    double elapsed_ms;
    double running_avg_gbps;
};
#pragma pack(pop)

struct complex_t {
    float r;
    float i;
};

static void send_frame(int fd, uint32_t npri, uint32_t nrg,
    const complex_t* raw_signal_ptr,
    const float* dpc_avg_ptr,
    const complex_t* fft_signal_ptr,
    const int* cfar_mask_ptr, const FrameStats& stats);

int main() {
    const char* fifo_path = "/tmp/radar_pipe";

    // 1. Create the named pipe (FIFO)
    // mkfifo returns -1 if it fails. If it fails because it already exists (EEXIST), that's fine.
    if (mkfifo(fifo_path, 0666) == -1 && errno != EEXIST) {
        std::cerr << "[C++] Error creating FIFO\n";
        return 1;
    }

    std::cout << "[C++] Waiting for Python Monitor to connect to the pipe..." << std::endl;

    // 2. Open the pipe.
    // NOTE: This open() call WILL BLOCK until the Python script opens the other end!
    int fd = open(fifo_path, O_WRONLY);
    if (fd == -1) {
        std::cerr << "[C++] Error opening FIFO for writing\n";
        return 1;
    }

    std::cout << "[C++] Connected! Starting data injection..." << std::endl;

    uint32_t npri = 128;
    uint32_t nrg = 256;
    int frame_count = 0;

    // Vectors to hold our memory
    std::vector<complex_t> raw_signal;
    std::vector<float> dpc_avg;
    std::vector<complex_t> fft_signal;
    std::vector<int> cfar_mask;

    while (true) {
        // Test the dynamic resizing feature: change dimensions every 20 frames
        if (frame_count % 40 < 20) {
            npri = 128; nrg = 256;
        } else {
            npri = 64; nrg = 128;
        }

        // Resize buffers if dimensions changed
        raw_signal.resize(npri * nrg);
        dpc_avg.resize(nrg);
        fft_signal.resize(npri * nrg);
        cfar_mask.resize(nrg);

        // 3. Generate Dummy Data
        for (uint32_t i = 0; i < nrg; ++i) {
            // Fake DPC wave
            dpc_avg[i] = std::sin(i * 0.1f + frame_count * 0.1f) * 10.0f;

            // Fake CFAR detections (spike every 50th gate)
            cfar_mask[i] = (i % 50 == 0) ? 1 : 0;

            for (uint32_t j = 0; j < npri; ++j) {
                uint32_t idx = j * nrg + i;

                // Add some moving noise patterns
                raw_signal[idx] = {
                    static_cast<float>(std::cos(j * 0.05 + frame_count * 0.05)),
                    static_cast<float>(std::sin(i * 0.05))
                };

                // Exaggerated FFT pattern
                fft_signal[idx] = {
                    static_cast<float>(std::cos(j * 0.1) * 5),
                    static_cast<float>(std::sin(i * 0.1) * 5)
                };
            }
        }

        FrameStats stats = {
            1.2 + (frame_count * 0.01), // throughput_gbps
            15.4,                       // elapsed_ms
            1.25                        // running_avg_gbps
        };

        // 4. Fire it into the pipe
        send_frame(fd, npri, nrg, raw_signal.data(), dpc_avg.data(), fft_signal.data(), cfar_mask.data(), stats);

        std::cout << "[C++] Sent frame " << frame_count << " (Size: " << npri << "x" << nrg << ")" << std::endl;
        frame_count++;

        // 5. Throttle the injection (e.g., 10 frames per second)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    close(fd);
    return 0;
}

void send_frame(const int fd, const uint32_t npri, const uint32_t nrg,
    const complex_t* raw_signal_ptr,
    const float* dpc_avg_ptr,
    const complex_t* fft_signal_ptr,
    const int* cfar_mask_ptr, const FrameStats& stats) {
    const FrameHeader header = {npri, nrg};

    // 1. Send the header (8 bytes)
    write(fd, &header, sizeof(FrameHeader));

    // 2. Send the dynamic payload
    write(fd, raw_signal_ptr, npri * nrg * sizeof(complex_t));
    write(fd, dpc_avg_ptr, nrg * sizeof(float));
    write(fd, fft_signal_ptr, npri * nrg * sizeof(complex_t));
    write(fd, cfar_mask_ptr, nrg * sizeof(int));
    write(fd, &stats, sizeof(FrameStats));
}