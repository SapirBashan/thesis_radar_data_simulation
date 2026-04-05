from online_monitor import OnlineMonitor
from pipe_monitor import PipeMonitor


def main():
    print("Initializing Radar Visualizer...")

    # 1. Create the UI Manager
    # show_every=1 means it will attempt to draw every frame it receives.
    visualizer = OnlineMonitor(show_every=1)

    # 2. Create the OS Pipe listener and bind it to the UI
    pipe_reader = PipeMonitor(
        monitor=visualizer,
        pipe_path="/tmp/radar_pipe"
    )

    # 3. Start the event loop (Blocks until Ctrl+C)
    pipe_reader.run()


if __name__ == "__main__":
    main()