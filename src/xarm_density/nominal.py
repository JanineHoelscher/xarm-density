"""The original OSCBF pose nominal with an explicit runtime reference."""

import jax
import jax.numpy as jnp
import numpy as np

from xarm_density._vendor.oscbf.controllers import PoseTaskVelocityController


class PoseReference:
    def __init__(self, robot, scene, q):
        self.duration = float(scene["reference_duration_s"])
        self.robot = robot
        self.start = np.asarray(robot.ee_position(jnp.asarray(q)))
        self.goal = np.asarray(scene["goal"], dtype=float)
        desired_q = jnp.asarray(q)
        rotation = robot.ee_transform(desired_q)[:3, :3]
        controller = PoseTaskVelocityController(
            n_joints=6, kp_task=np.r_[[scene["kp_position"]]*3, [scene["kp_orientation"]]*3],
            kp_joint=.2, qdot_min=-np.ones(6)*scene["qdot_limit"],
            qdot_max=np.ones(6)*scene["qdot_limit"],
        )
        @jax.jit
        def command(q, position, velocity):
            m_inv, jacobian, transform = robot.dynamically_consistent_velocity_control_matrices(q)
            return controller(q, transform[:3, 3], transform[:3, :3], position,
                              rotation, velocity, jnp.zeros(3), desired_q, jacobian, m_inv)
        self.kernel = command

    def reference(self, t):
        s = np.clip(t / self.duration, 0., 1.)
        blend = 10*s**3 - 15*s**4 + 6*s**5
        rate = (30*s**2 - 60*s**3 + 30*s**4) / self.duration
        delta = self.goal - self.start
        return self.start + blend*delta, rate*delta

    def command(self, q, t):
        position, velocity = self.reference(t)
        return np.asarray(self.kernel(jnp.asarray(q), jnp.asarray(position), jnp.asarray(velocity)))
