"""UF850 MoveIt Servo with fake ros2_control hardware and CDF velocity input."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    fake_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('xarm_moveit_servo'), 'launch',
            'uf850_moveit_servo_fake.launch.py'])),
        launch_arguments={'add_gripper': 'true'}.items(),
    )
    controller = ExecuteProcess(
        cmd=[EnvironmentVariable('XARM_DENSITY_PYTHON', default_value='python3'),
             '-m', 'xarm_density_ros.cdf_velocity_controller'],
        output='screen',
    )
    return LaunchDescription([fake_moveit, controller])
