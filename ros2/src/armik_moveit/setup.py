from glob import glob

from setuptools import find_packages, setup

package_name = "armik_moveit"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mohamed Kamel",
    maintainer_email="mkamel860@gmail.com",
    description="ROS 2 + MoveIt 2 port of the robot-arm-ik palletizing cell (Phase 2).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "plan_execute_smoke = armik_moveit.plan_execute_smoke:main",
        ],
    },
)
