#!/usr/bin/env python3
"""Test client for Isaac Sim HTTP server."""

import requests
import time
import math
import base64
from pathlib import Path

import cv2
import numpy as np

BASE_URL = "http://localhost:8770"
OUTPUT_DIR = Path(__file__).resolve().parent


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


def test_spawn_object():
    """Test spawning/resetting the simulated object."""
    print("\nTesting POST /object...")

    payload = {
        "type": "block",
        "position": [0.0, 0.15, 0.02],
        "rotation_rad": math.pi / 6,
    }
    print(f"  Spawning object: {payload}")
    resp = requests.post(f"{BASE_URL}/object", json=payload)
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Response: {data}")

    assert resp.status_code == 200
    assert data["ok"] is True
    assert data["name"] == "cube"
    assert len(data["position"]) == 3
    return payload


def test_get_object_pose(expected_position=None):
    """Test querying the simulated object position."""
    print("\nTesting GET /object/pose...")

    resp = requests.get(f"{BASE_URL}/object/pose")
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Response: {data}")

    assert resp.status_code == 200
    assert data["ok"] is True
    assert "position" in data
    assert len(data["position"]) == 3

    if expected_position is not None:
        position = np.array(data["position"], dtype=np.float32)
        expected = np.array(expected_position, dtype=np.float32)
        distance = np.linalg.norm(position - expected)
        print(f"  Distance from expected: {distance:.4f} m")
        assert distance < 0.05

    return data


def test_get_camera_rgb(camera_name="gripper_cam", output_path=None):
    """Test querying and decoding the gripper RGB camera frame."""
    if output_path is None:
        output_path = OUTPUT_DIR / f"http_camera_rgb_{camera_name}.png"
    print(f"\nTesting GET /camera/rgb?camera={camera_name}...")

    resp = requests.get(f"{BASE_URL}/camera/rgb", params={"camera": camera_name})
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(
        "  Response metadata: "
        f"ok={data.get('ok')}, camera={data.get('camera')}, "
        f"encoding={data.get('encoding')}, shape={data.get('shape')}"
    )

    assert resp.status_code == 200
    assert data["ok"] is True
    assert data["encoding"] == "png_base64"
    assert "image" in data

    raw = base64.b64decode(data["image"])
    image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    expected_shape = data.get("shape")
    if expected_shape is not None:
        assert list(image_rgb.shape) == expected_shape

    cv2.imwrite(str(output_path), image_bgr)
    print(f"  Decoded image shape: {image_rgb.shape}, dtype={image_rgb.dtype}")
    print(f"  Saved image: {output_path}")
    return image_rgb


def test_get_all_camera_rgbs():
    """Test and save all camera RGB views."""
    images = {}
    for camera_name in ("gripper_cam", "top", "side", "front"):
        images[camera_name] = test_get_camera_rgb(camera_name)
    return images


def test_unknown_camera_returns_error():
    """Test unknown camera names return a structured error response."""
    print("\nTesting GET /camera/rgb with an unknown camera...")

    resp = requests.get(f"{BASE_URL}/camera/rgb", params={"camera": "missing_camera"})
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Response: {data}")

    assert resp.status_code == 400
    assert data["ok"] is False
    assert "available" in data
    return data


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
        object_payload = test_spawn_object()
        test_get_object_pose(object_payload["position"])
        test_get_all_camera_rgbs()
        test_unknown_camera_returns_error()

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
