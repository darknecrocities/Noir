"""Main entrypoint for Project NOIR."""

import argparse
import sys
import time
# Ensure torch and core subsystems load before PySide6 to avoid Shiboken hook inspection latency
import torch
from noir.core.engine import NoirEngine
from noir.core.logging import setup_logging


def main() -> None:
    """Project NOIR launcher."""
    parser = argparse.ArgumentParser(description="PROJECT NOIR — Real-Time AI & ML Research Environment")
    parser.add_argument("--config", type=str, default=None, help="Path to custom YAML configuration file")
    parser.add_argument("--headless", action="store_true", help="Run in headless CLI mode without desktop GUI")
    parser.add_argument("--mode", type=str, choices=["all", "autonomous", "supervised", "rl", "llm", "open_web"], default=None, help="Directly start experiment in specified mode")
    parser.add_argument("--recover", action="store_true", help="Automatically recover previous session on launch")
    args = parser.parse_args()

    # 1. Initialize Structured Logging
    setup_logging(log_dir="logs", log_level="INFO")

    # 2. Instantiate Master Noir Engine
    engine = NoirEngine(config_path=args.config)
    engine.start()

    def handle_interrupt(sig=None, frame=None):
        print("\n[SHUTDOWN TRIGGERED] Ctrl+C / Interrupt received.")
        print("[AUTO-SAVE] Automatically saving active neural weights and training state...")
        try:
            engine.shutdown()
            print("[AUTO-SAVE COMPLETED] Model checkpoint preserved. Resume anytime with: python -m noir.main --recover")
        except Exception as e:
            print(f"[SHUTDOWN NOTICE] {e}")
        sys.exit(0)

    # 3. Headless Execution Mode
    if args.headless:
        print("==========================================================")
        print("        PROJECT NOIR — AUTONOMOUS RESEARCH RUNNER         ")
        print("==========================================================")

        if args.recover:
            engine.recover_from_previous_session(action="resume")
        elif args.mode in ("supervised", "digits"):
            engine.start_supervised_experiment(dataset_name="digits", name="Real Digits Headless Run")
        elif args.mode == "rl":
            engine.start_rl_experiment(name="PPO GridWorld Headless Run")
        else:
            # Default: Start autonomous master research loop
            engine.start_autonomous_master_training()

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            handle_interrupt()

    # 4. Native Desktop GUI Mode
    from PySide6.QtWidgets import QApplication
    from noir.ui.main_window import NoirMainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Project NOIR")
    app.setStyle("Fusion")

    window = NoirMainWindow(engine=engine)
    window.show()

    # Auto-recovery check
    if args.recover:
        engine.recover_from_previous_session(action="resume")
    else:
        window.check_startup_recovery()

    if args.mode in ("all", "autonomous"):
        engine.start_autonomous_master_training()
    elif args.mode in ("llm", "open_web"):
        engine.start_open_web_llm_experiment()
    elif args.mode == "supervised":
        engine.start_supervised_experiment()
    elif args.mode == "rl":
        engine.start_rl_experiment()

    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        handle_interrupt()
    finally:
        engine.shutdown()
        sys.exit(exit_code if "exit_code" in locals() else 0)


if __name__ == "__main__":
    main()
