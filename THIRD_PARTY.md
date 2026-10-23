# Included code and robot assets

## OSCBF numerical subset

`src/xarm_density/_vendor/oscbf/{manipulator,controllers}.py` comes from
https://github.com/StanfordASL/oscbf at
`6082329f8e1d61ee8e875a63a5cd80a14d513e12`, copyright Daniel Morton (2025),
under the included MIT [license](src/xarm_density/_vendor/oscbf/LICENSE).
The manipulator import uses the bundled checked descriptor rather than an
online URDF parser. Unused Franka entry points and asset imports were removed.
The numerical algorithms and nominal controller are retained.

The local CDF-QP matrix assembly and smooth obstacle window were brought into
this project from the author's `density-drift` implementation. There is no
runtime import or filesystem dependency on that checkout.

## UFACTORY model

`src/xarm_density/assets/uf850/` includes only referenced UF850/gripper meshes
and expanded URDFs derived from https://github.com/xArm-Developer/xarm_ros2 at
`d0b95117dabd3883f41155125aa3f67d37901c18`. The included UFACTORY BSD notice is
in [LICENSE](src/xarm_density/assets/uf850/LICENSE). The original vendor copyright
and redistribution conditions apply to these model assets.

Mesh paths are relative. ROS control and Gazebo plugin blocks were removed
from the simulation URDFs. The arm descriptor was produced using the original
OSCBF parser before relocation; it preserves the tested mass, joint, and
fixed-link-merging data. Its URDF hash is checked at load time. The package
supports this bundled robot, not arbitrary URDF replacement without a new
validated descriptor. The gripper's geometry is included for collision checks;
the original arm-only nominal mass model is unchanged.
