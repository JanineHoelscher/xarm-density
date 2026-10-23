"""Load the bundled UF850 model descriptor without ROS or a mesh parser."""

import hashlib
import json
from pathlib import Path


def parse_urdf(filename):
    """Load the checked descriptor produced from the matching vendor URDF.

    Only models shipped with a matching descriptor are supported. This keeps
    the tested fixed-link merging and inertias without an online URDF parser.
    """
    path = Path(filename)
    payload = json.loads(path.with_suffix(".kinematics.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != payload["urdf_sha256"]:
        raise ValueError("Robot URDF differs from its bundled kinematics descriptor")
    return payload["kinematics"]
