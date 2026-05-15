#!/usr/bin/env python3
"""HTTP server for controlling Isaac Sim robot simulation.

Provides REST API endpoints to control the robot and query its state.
"""

import sys
import threading
import time
import math
import torch
from flask import Flask, request, jsonify

# Parse HTTP server arguments before importing isaac_koch
host = "0.0.0.0"
port = 8770

# Extract --host and --port from command line args
args_to_remove = []
for i, arg in enumerate(sys.argv):
    if arg == "--host" and i + 1 < len(sys.argv):
        host = sys.argv[i + 1]
        args_to_remove.extend([i, i + 1])
    elif arg == "--port" and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])
        args_to_remove.extend([i, i + 1])

# Remove HTTP server args so isaac_koch doesn't see them
for i in sorted(args_to_remove, reverse=True):
    sys.argv.pop(i)

# Import isaac_koch (this will initialize AppLauncher)
from isaac_koch import (
    SimIsaacModel,
    URDF_PATH,
    JOINT_NAMES,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    simulation_app,
)

# Flask app
app = Flask(__name__)

# Global state
sim = None
state_lock = threading.Lock()  # Protects sim state reads
shutdown_flag = threading.Event()
torque_enabled = True

# Command queue: pending joint angle command from HTTP handlers,
# consumed by the main simulation loop.
# Stored as list of 6 floats (radians): [joint1..joint5, gripper]
pending_command = None
command_lock = threading.Lock()


def gripper_angle_to_0to1(angle: float) -> float:
    """Convert gripper angle (radians) to 0-1 range.

    Args:
        angle: Gripper angle in sim definition (GRIPPER_CLOSED=0.0 to GRIPPER_OPEN=-1.0)

    Returns:
        float: 0.0 (closed) to 1.0 (open)
    """
    return (angle - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)


def gripper_0to1_to_angle(value: float) -> float:
    """Convert 0-1 range to gripper angle (radians).

    Args:
        value: 0.0 (closed) to 1.0 (open)

    Returns:
        float: Gripper angle in sim definition
    """
    return GRIPPER_CLOSED + value * (GRIPPER_OPEN - GRIPPER_CLOSED)


def flask_server():
    """Run Flask server in background thread."""
    print(f"[Server] Starting HTTP server on {host}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"ok": True})


@app.route("/state", methods=["GET"])
def get_state():
    """Get current robot state."""
    try:
        with state_lock:
            if sim is None:
                return (
                    jsonify({"ok": False, "error": "Simulation not initialized"}),
                    503,
                )

            joint_angles = sim.get_joint_angles()
            gripper_angle = joint_angles[5]

            # Convert to degrees
            joint_angles_deg = [math.degrees(a) for a in joint_angles[:5]]
            gripper_0to1 = gripper_angle_to_0to1(gripper_angle)

            return jsonify(
                {
                    "joint_names": JOINT_NAMES[:5],
                    "joint_angles_deg": joint_angles_deg,
                    "gripper_open_0to1": gripper_0to1,
                }
            )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/state", methods=["POST"])
def set_state():
    """Set robot state.

    Records the desired joint targets into a command queue. The main
    simulation loop consumes the queue and applies the targets via
    sim.set_joint_angles().
    """
    global pending_command

    try:
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "No JSON data provided"}), 400

        if sim is None:
            return (
                jsonify({"ok": False, "error": "Simulation not initialized"}),
                503,
            )

        # Read current joint angles to use as defaults for unspecified fields.
        with state_lock:
            current_angles = sim.get_joint_angles()

        # Validate and convert joint angles
        if "joint_angles_deg" in data:
            joint_angles_deg = data["joint_angles_deg"]
            if not isinstance(joint_angles_deg, list) or len(joint_angles_deg) != 5:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "joint_angles_deg must be list of 5 floats",
                        }
                    ),
                    400,
                )
            joint_angles_rad = [math.radians(a) for a in joint_angles_deg]
        else:
            joint_angles_rad = list(current_angles[:5])

        # Validate and convert gripper
        if "gripper_open_0to1" in data:
            gripper_0to1 = data["gripper_open_0to1"]
            if not isinstance(gripper_0to1, (int, float)) or not (
                0 <= gripper_0to1 <= 1
            ):
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "gripper_open_0to1 must be float in [0, 1]",
                        }
                    ),
                    400,
                )
            gripper_angle = gripper_0to1_to_angle(gripper_0to1)
        else:
            gripper_angle = current_angles[5]

        # Enqueue command for main thread (overwrites any pending command)
        with command_lock:
            pending_command = joint_angles_rad + [gripper_angle]

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/torque", methods=["POST"])
def set_torque():
    """Enable or disable torque control."""
    global torque_enabled

    try:
        data = request.get_json()
        if not data or "enabled" not in data:
            return jsonify({"ok": False, "error": "Missing 'enabled' field"}), 400

        enabled = data["enabled"]
        if not isinstance(enabled, bool):
            return jsonify({"ok": False, "error": "'enabled' must be boolean"}), 400

        torque_enabled = enabled
        print(f"[Server] Torque {'enabled' if enabled else 'disabled'}")

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shutdown the server and simulation."""
    try:
        print("[Server] Shutdown requested")
        shutdown_flag.set()

        # Shutdown Flask server
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def main():
    """Main entry point."""
    global sim, pending_command

    print(f"[Server] Initializing simulation...")

    # Initialize simulation in main thread
    sim = SimIsaacModel(URDF_PATH)
    print("[Server] Simulation initialized")

    # Start Flask server in background thread
    flask_thread = threading.Thread(target=flask_server, daemon=True)
    flask_thread.start()

    print("[Server] HTTP server started, entering simulation loop")

    # Run simulation loop in main thread
    try:
        while not shutdown_flag.is_set() and simulation_app.is_running():
            # Consume pending command from HTTP handlers
            cmd = None
            with command_lock:
                if pending_command is not None:
                    cmd = pending_command
                    pending_command = None

            with state_lock:
                if cmd is not None:
                    # set_joint_angles internally calls sim.step()
                    sim.set_joint_angles(cmd)
                elif torque_enabled:
                    sim.step()
                else:
                    # Step simulation but don't apply joint targets
                    sim._scene.write_data_to_sim()
                    sim._sim.step()
                    sim._scene.update(sim._sim_dt)
    except KeyboardInterrupt:
        print("\n[Server] Keyboard interrupt received")
    finally:
        print("[Server] Shutting down...")
        shutdown_flag.set()
        simulation_app.close()
        print("[Server] Shutdown complete")


if __name__ == "__main__":
    main()
