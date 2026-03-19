from setuptools import find_packages, setup

package_name = 'perception_suite'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shen6',
    maintainer_email='felixshn@umich.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'video_node = perception_suite.video_node:main',
        'inference_node = perception_suite.inference_node:main',
        'visualizer_node = perception_suite.visualizer_node:main',
    ],
},
)
