"""Run the UF850 whole-arm CDF-QP simulation.

Quick start (from the repository root):
    conda activate xarm
    python run_simulation.py

Examples:
    python run_simulation.py --headless
    python run_simulation.py --config my_experiment.json --hz 50
    python run_simulation.py --goal 0.38 0.35 0.43 --output results/my_run

Experiment parameters live in src/xarm_density/assets/scene.json. This script
shows the experiment workflow; the reusable helpers are in xarm_density.cli.
The actual nominal-control -> CDF-QP -> joint integration step is implemented
by Simulation.step() in src/xarm_density/simulation.py. No hardware is commanded.
"""

try:
    from xarm_density.cli import (
        print_run_settings,
        print_summary,
        run_cli,
        run_control_loop,
        save_results,
        simulation_session,
    )
except ModuleNotFoundError as error:
    if error.name != "xarm_density":
        raise
    raise SystemExit(
        "The project is not installed in this Python environment.\n"
        "From the repository root, run:\n"
        "  conda activate xarm\n"
        "  python -m pip install -e .\n"
    ) from None


def run_experiment(config, options):
    """Execute one experiment with validated config and command-line options."""
    # Display the effective settings, including any command-line overrides.
    print_run_settings(config, options)

    # Load the bundled UF850/gripper and compile the controller. The context
    # closes PyBullet and the optional ROS socket even when the run fails.
    with simulation_session(config, options) as (simulation, ros_bridge):
        try:
            # Each step computes a nominal pose-tracking velocity, filters it
            # through the CDF-QP, integrates joints, and checks clearances.
            run_control_loop(simulation, config, options, ros_bridge)
        except (KeyboardInterrupt, ValueError, RuntimeError, OSError):
            save_results(simulation, config, options.output, status="stopped")
            raise

        # Save the effective config, metrics, and every recorded trajectory row.
        summary = save_results(simulation, config, options.output)
        print_summary(summary, options.output)


def main(argv=None):
    # Shared argument parsing, numerical setup, validation, and error messages.
    # xarm-sim uses the same helpers and control loop as this readable example.
    return run_cli(argv=argv, workflow=run_experiment)


if __name__ == "__main__":
    raise SystemExit(main())
