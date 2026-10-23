from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    model = Path(get_package_share_directory('xarm_density_ros')) / 'model'
    robot = ET.parse(model / 'gripper.urdf')
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename', (model / mesh.get('filename')).resolve().as_uri())
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': ET.tostring(robot.getroot(), encoding='unicode')}]),
        Node(package='xarm_density_ros', executable='simulation_bridge', output='screen'),
    ])
