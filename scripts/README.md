# xArm / UF850 Scripts

This folder contains small standalone experiments for the UF850 planner work.

## OSCBF PyBullet Playground

The main ROS-free playground is:

```bash
/home/sriramk/xarm_ws/scripts/uf850_oscbf_pybullet_playground.py
```

It uses:

- the local UF850 URDF/xacro from `xarm_ros2`;
- PyBullet for visualization;
- StanfordASL `oscbf` and `cbfpy`;
- a nominal operational-space velocity controller;
- an OSCBF velocity safety filter.

It does not use ROS, MoveIt, or Gazebo.

## Environment Setup

Use the existing conda environment named `xarm`.

From any terminal:

```bash
source /home/sriramk/anaconda3/etc/profile.d/conda.sh
conda activate xarm
```

If the dependencies need to be installed or refreshed:

```bash
conda env update -n xarm -f /home/sriramk/xarm_ws/envs/xarm_oscbf_pybullet_environment.yml --prune
```

Quick dependency check:

```bash
python -c "import oscbf, cbfpy, pybullet, jax; print('ok')"
```

## Run

First run, regenerate the UF850 URDF:

```bash
python /home/sriramk/xarm_ws/scripts/uf850_oscbf_pybullet_playground.py --regen-urdf --num-obs 12
```

After the URDF has been generated once:

```bash
python /home/sriramk/xarm_ws/scripts/uf850_oscbf_pybullet_playground.py --num-obs 12
```

Show the xArm gripper in PyBullet:

```bash
python /home/sriramk/xarm_ws/scripts/uf850_oscbf_pybullet_playground.py --regen-urdf --with-gripper --num-obs 12
```

`--with-gripper` is visualization-only for now: OSCBF still parses an arm-only URDF and controls the six UF850 arm joints. This avoids treating gripper mimic joints as extra planner decision variables.

Headless smoke test:

```bash
python /home/sriramk/xarm_ws/scripts/uf850_oscbf_pybullet_playground.py --no-gui --steps 20 --num-obs 1
```

Try a different start, goal, and obstacle count:

```bash
python /home/sriramk/xarm_ws/scripts/uf850_oscbf_pybullet_playground.py \
  --start-q "0,-0.7,-1.0,0,0.8,0" \
  --goal "0.55,0.15,0.45" \
  --num-obs 20 \
  --seed 4 \
  --qdot-limit 0.7
```

Useful knobs:

- `--start-q`: six comma-separated joint values.
- `--goal`: desired end-effector position, `x,y,z`.
- `--num-obs`: number of random spherical obstacles.
- `--seed`: obstacle random seed.
- `--kp-task`: task-space proportional gain.
- `--kp-posture`: null-space posture gain.
- `--alpha`: CBF class-K gain.
- `--qdot-limit`: joint velocity limit used by the QP.
- `--safety-margin`: extra margin added to robot collision spheres.
- `--radius-scale`: scales the rough UF850 link-sphere model.
- `--with-gripper`: load an arm+gripper URDF in PyBullet while keeping OSCBF control arm-only.

## Notes

The current UF850 link collision model is a rough manually defined sphere
decomposition. It is enough for controller experimentation, but not final robot
safety validation.

The generated URDF is written to:

```bash
/home/sriramk/xarm_ws/generated/uf850_oscbf_pybullet.urdf
```

With `--with-gripper`, the PyBullet visualization URDF is written to:

```bash
/home/sriramk/xarm_ws/generated/uf850_oscbf_pybullet_gripper.urdf
```

