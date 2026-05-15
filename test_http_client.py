#!/usr/bin/env python3
"""Test client for Isaac Sim HTTP server."""

import requests
import time
import math

BASE_URL = "http://localhost:8770"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    assert resp.json()["ok"] is True


def test_get_state():
    """Test getting robot state."""
    print("\nTesting GET /state...")
    resp = requests.get(f"{BASE_URL}/state")
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Joint names: {data['joint_names']}")
    print(f"  Joint angles (deg): {data['joint_angles_deg']}")
    print(f"  Gripper open (0-1): {data['gripper_open_0to1']}")
    return data


def test_set_state():
    """Test setting robot state."""
    print("\nTesting POST /state...")

    # Move joints to specific angles
    payload = {
        "joint_angles_deg": [
            -10.00012108759119869573,
            0.27044654286981173,
            -0.18145744891674365,
            0.0010321537664444963,
            0.0008376255265863699,
        ],
        "gripper_open_0to1": 0.5,
    }
    print(f"  Setting state: {payload}")
    resp = requests.post(f"{BASE_URL}/state", json=payload)
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")

    # Wait for motion
    time.sleep(1)

    # Verify state changed
    state = test_get_state()
    print(f"  Verified new state: {state['joint_angles_deg']}")


def test_gripper():
    """Test gripper control."""
    print("\nTesting gripper control...")

    # Open gripper
    print("  Opening gripper...")
    resp = requests.post(f"{BASE_URL}/state", json={"gripper_open_0to1": 1.0})
    assert resp.json()["ok"] is True
    time.sleep(1)

    state = requests.get(f"{BASE_URL}/state").json()
    print(f"  Gripper state: {state['gripper_open_0to1']:.3f} (should be ~1.0)")

    # Close gripper
    print("  Closing gripper...")
    resp = requests.post(f"{BASE_URL}/state", json={"gripper_open_0to1": 0.0})
    assert resp.json()["ok"] is True
    time.sleep(1)

    state = requests.get(f"{BASE_URL}/state").json()
    print(f"  Gripper state: {state['gripper_open_0to1']:.3f} (should be ~0.0)")


def test_torque():
    """Test torque enable/disable."""
    print("\nTesting POST /torque...")

    # Disable torque
    print("  Disabling torque...")
    resp = requests.post(f"{BASE_URL}/torque", json={"enabled": False})
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    time.sleep(1)

    # Enable torque
    print("  Enabling torque...")
    resp = requests.post(f"{BASE_URL}/torque", json={"enabled": True})
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")


def test_shutdown():
    """Test shutdown endpoint."""
    print("\nTesting POST /shutdown...")
    resp = requests.post(f"{BASE_URL}/shutdown", json={})
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Isaac Sim HTTP Server Test Client")
    print("=" * 60)

    try:
        test_health()
        test_get_state()
        test_set_state()
        test_gripper()
        test_torque()

        # Uncomment to test shutdown
        # test_shutdown()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to server.")
        print("Make sure the server is running:")
        print("  python isaac_http_server.py")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
