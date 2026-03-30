import time

from online_monitor import OnlineMonitor


def main() -> None:
    monitor = OnlineMonitor(output_dir="output_signal", show_every=1)
    try:
        step = 0
        while True:
            monitor.update(step)
            step += 1
            time.sleep(1.25)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
