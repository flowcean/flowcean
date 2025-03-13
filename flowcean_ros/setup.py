from setuptools import setup

package_name = "flowcean_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "polars", "numpy"],
    zip_safe=True,
    maintainer="Markus Knitt",
    maintainer_email="markus.knitt@tuhh.de",
    description="ROS 2 package for flowcean-based ML models",
    license='BSD 3-Clause "New" or "Revised" License',
    entry_points={
        "console_scripts": [
            "data_preprocessor = flowcean_ros.data_preprocessor:main",
        ],
    },
)
