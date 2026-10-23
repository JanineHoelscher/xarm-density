from pathlib import Path
import os
from setuptools import setup

root = Path(__file__).resolve().parents[2]
assets = root / 'src/xarm_density/assets/uf850'
data = [('share/ament_index/resource_index/packages', ['resource/xarm_density_ros']),
        ('share/xarm_density_ros', ['package.xml']),
        ('share/xarm_density_ros', [os.path.relpath(
            root / 'src/xarm_density/assets/scene.json', Path(__file__).resolve().parent)]),
        ('share/xarm_density_ros/launch', [
            'launch/simulation.launch.py', 'launch/fake_moveit.launch.py'])]
for directory in sorted({p.parent for p in assets.rglob('*') if p.is_file()}):
    files = [os.path.relpath(p, Path(__file__).resolve().parent)
             for p in directory.iterdir() if p.is_file()]
    data.append((str(Path('share/xarm_density_ros/model') / directory.relative_to(assets)), files))
setup(name='xarm_density_ros', version='0.2.0', packages=['xarm_density_ros'],
      tests_require=['pytest'],
      data_files=data, install_requires=['setuptools'], zip_safe=False,
      maintainer='Sriram', maintainer_email='sriramk@clemson.edu',
      description='Simulation-only density controller ROS bridge', license='BSD-3-Clause',
      entry_points={'console_scripts': ['simulation_bridge = xarm_density_ros.node:main']})
