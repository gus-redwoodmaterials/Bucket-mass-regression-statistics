#!/usr/bin/env python3
"""
Controller Response Simulation - ACME Thermal Oxidizer O2 Controller

This script simulates the ACME thermal oxidizer O2 controller's response to a fake signal
that dips from 15% to 5% at 1%/s and plots the fresh air valve percent open response.

Based on the actual PD controller implementation from thermal_oxidizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PDController:
    """PD controller implementation based on ACME thermal oxidizer O2 controller"""

    def __init__(
        self,
        o2_normal_percent=12.0,
        o2_min_percent=5.0,
        o2_p_scaling=1.0,
        o2_d_gain=-5.0,
        controller_run_interval=2.0,
        d_term_lookback_cycles=4.0,
    ):
        # Constants from your actual thermal oxidizer implementation
        self.o2_normal_percent = o2_normal_percent  # Normal O2 setpoint (12%)
        self.o2_min_percent = o2_min_percent  # Min O2 setpoint (5%)
        self.o2_p_scaling = o2_p_scaling  # P gain scaling factor
        self.o2_d_gain = o2_d_gain  # D gain (-5% valve per %O2/sec)
        self.controller_run_interval = controller_run_interval  # Controller cycle time (2s)
        self.d_term_lookback_cycles = d_term_lookback_cycles  # Lookback cycles for D term (4)

        # Calculate P gain like in your actual controller
        self.o2_p_gain = self.o2_p_scaling / (self.o2_normal_percent - self.o2_min_percent) * 100

        # History storage for derivative calculation
        self.o2_history = []
        self.time_history = []

    def update(self, current_o2, current_time):
        """
        Update PD controller based on ACME thermal oxidizer O2 controller logic

        Returns fresh air valve percentage (0-100%)
        """
        # Store history for derivative calculation
        self.o2_history.append(current_o2)
        self.time_history.append(current_time)

        # Limit history length based on lookback time
        d_term_lookback_s = self.d_term_lookback_cycles * self.controller_run_interval
        max_history_points = int(d_term_lookback_s / 0.1) + 10  # +10 for safety margin

        if len(self.o2_history) > max_history_points:
            self.o2_history = self.o2_history[-max_history_points:]
            self.time_history = self.time_history[-max_history_points:]

        # ACME controller logic: Handle edge cases first
        if current_o2 is None or current_o2 < 0:
            return 0  # Fresh air valve closed

        if current_o2 > self.o2_normal_percent:
            # O2 above normal - no adjustment needed
            return 0

        if current_o2 <= self.o2_min_percent:
            # O2 below minimum - fully open fresh air valve
            return 100

        # Calculate P term (proportional to error from normal O2)
        err = self.o2_normal_percent - current_o2  # Error from normal setpoint
        p_term = err * self.o2_p_gain

        # Calculate D term (derivative of O2 over time)
        d_term = 0
        if len(self.o2_history) >= 2:
            # Find data points from lookback time ago
            lookback_time = current_time - d_term_lookback_s

            # Find the closest historical point to our lookback time
            historical_idx = 0
            for i, t in enumerate(self.time_history):
                if t >= lookback_time:
                    historical_idx = i
                    break

            if historical_idx < len(self.o2_history) - 1:
                # Calculate derivative: dO2/dt
                d_o2 = self.o2_history[-1] - self.o2_history[historical_idx]
                dt = self.time_history[-1] - self.time_history[historical_idx]

                if dt > 0:
                    do2_dt = d_o2 / dt  # %O2/sec
                    d_term = do2_dt * self.o2_d_gain  # Apply D gain

        # Combine P and D terms
        o2_control_term = p_term + d_term

        # Clamp to 0-100% valve opening (like in ACME controller)
        o2_control_term = max(0.0, min(100.0, o2_control_term))

        return o2_control_term


def simulate_o2_signal(time_array):
    """
    Generate the fake O2 signal that dips from 15% to 5% at 1%/s

    Args:
        time_array: Array of time points

    Returns:
        Array of O2 values
    """
    o2_values = np.full_like(time_array, 15.0)  # Start at 15%

    # Find when the dip starts and ends
    dip_start_time = 20.0  # Start dip at t=20s
    dip_duration = 10.0  # 10% change at 1%/s = 10 seconds
    dip_end_time = dip_start_time + dip_duration
    recovery_duration = 15.0  # Take 15 seconds to recover back to 15%
    recovery_end_time = dip_end_time + recovery_duration

    for i, t in enumerate(time_array):
        if dip_start_time <= t <= dip_end_time:
            # Linear dip from 15% to 5% at 1%/s
            progress = (t - dip_start_time) / dip_duration
            o2_values[i] = 15.0 - 10.0 * progress
        elif dip_end_time < t <= recovery_end_time:
            # Recovery back to 15%
            progress = (t - dip_end_time) / recovery_duration
            o2_values[i] = 5.0 + 10.0 * progress
        elif t > recovery_end_time:
            o2_values[i] = 15.0

    return o2_values


def run_simulation():
    """Run the ACME thermal oxidizer O2 controller response simulation"""

    # Simulation parameters
    total_time = 60.0  # Total simulation time (seconds)
    dt = 0.1  # Time step (seconds)
    time_array = np.arange(0, total_time + dt, dt)

    # Initialize controller (based on ACME thermal oxidizer O2 controller)
    controller = PDController(
        o2_normal_percent=12.0,  # Normal O2 setpoint
        o2_min_percent=5.0,  # Min O2 setpoint
        o2_p_scaling=1.0,  # P gain scaling
        o2_d_gain=-5.0,  # D gain (valve % per %O2/sec)
        controller_run_interval=2.0,  # Controller cycle time
        d_term_lookback_cycles=4.0,  # Lookback cycles for D term
    )

    # Initialize arrays for storing results
    o2_signal = simulate_o2_signal(time_array)
    valve_position = np.zeros_like(time_array)
    controller_output = np.zeros_like(time_array)

    # Initial conditions (valve starts at minimum fresh air opening)
    valve_position[0] = 0.0  # Start valve at minimum (fresh air min valve percent)

    # Run simulation
    for i in range(1, len(time_array)):
        # Get current O2 reading and time
        current_o2 = o2_signal[i]
        current_time = time_array[i]

        # Update controller (returns fresh air valve percentage directly)
        valve_position[i] = controller.update(current_o2, current_time)

        # Store controller output for analysis (valve position is the output)
        controller_output[i] = valve_position[i]

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "time": time_array,
            "o2_signal": o2_signal,
            "valve_position": valve_position,
            "controller_output": controller_output,
            "error": controller.o2_normal_percent - o2_signal,  # Error from normal O2 setpoint
        }
    )

    return results


def plot_results(results):
    """Plot the simulation results"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot O2 signal and fresh air valve response on same plot
    ax.plot(results["time"], results["o2_signal"], "b--", linewidth=2, label="O2 Signal (%)")
    ax.plot(results["time"], results["valve_position"], "g-", linewidth=2, label="Fresh Air Valve Position (%)")

    ax.set_ylabel("Valve % Open", fontsize=14)
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_title("O2 Signal & Fresh Air Valve Response - ACME O2 Control", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print(f"O2 Signal Range: {results['o2_signal'].min():.1f}% - {results['o2_signal'].max():.1f}%")
    print(f"Valve Position Range: {results['valve_position'].min():.1f}% - {results['valve_position'].max():.1f}%")
    print(
        f"Maximum Valve Response: {results['valve_position'].max() - results['valve_position'].min():.1f}% total swing"
    )
    print(
        f"Controller Output Range: {results['controller_output'].min():.2f} - {results['controller_output'].max():.2f}"
    )

    # Find key response times
    min_o2_idx = np.argmin(results["o2_signal"])
    max_valve_idx = np.argmax(results["valve_position"])

    print("\nKey Response Times:")
    print(f"O2 minimum ({results['o2_signal'].iloc[min_o2_idx]:.1f}%) at t={results['time'].iloc[min_o2_idx]:.1f}s")
    print(
        f"Max valve opening ({results['valve_position'].iloc[max_valve_idx]:.1f}%) at t={results['time'].iloc[max_valve_idx]:.1f}s"
    )

    return results


def save_results(results, filename="controller_simulation_results.csv"):
    """Save simulation results to CSV"""
    filepath = f"/Users/gus.robinson/Desktop/Local Python Coding/{filename}"
    results.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    print("🎛️  ACME THERMAL OXIDIZER O2 CONTROLLER SIMULATION")
    print("Simulating fresh air valve response to O2 signal dip (15% → 5% @ 1%/s)")
    print("Based on actual ACME thermal oxidizer PD controller implementation")
    print("-" * 70)

    # Run the simulation
    results = run_simulation()

    # Plot the results
    plot_results(results)

    # Save results
    save_results(results)

    print("\nSimulation complete! 🎯")
