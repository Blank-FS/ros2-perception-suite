"""Starts the whole pipeline: one source node, then stages 2-6.

    ros2 launch perception_suite pipeline.launch.py
    ros2 launch perception_suite pipeline.launch.py source:=video video_path:=other.mp4
    ros2 launch perception_suite pipeline.launch.py source:=camera
    ros2 launch perception_suite pipeline.launch.py source:=camera camera_index:=0
    ros2 launch perception_suite pipeline.launch.py scene:=snow
    ros2 launch perception_suite pipeline.launch.py visualize:=false
    ros2 launch perception_suite pipeline.launch.py backend:=tensorrt

Run it from the workspace root, where the relative video and weight paths
point.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Source name -> (executable, launch arguments it takes as parameters).
# A new input (camera, beamng) is one entry here plus its node.
SOURCES = {
    'video': ('1_video_node', ['video_path', 'loop', 'video_fps']),
    'camera': ('1_camera_node', ['camera_index', 'camera_fps']),
}


def _scene_prompts(context):
    config_path = LaunchConfiguration('config').perform(context)
    with open(config_path, encoding='utf-8') as fh:
        raw = yaml.safe_load(fh) or {}
    scenes = raw.get('perception', {}).get('scenes', {})
    name = LaunchConfiguration('scene').perform(context)
    if name not in scenes:
        raise ValueError(
            f"scene must be one of {sorted(scenes)} (perception.scenes in {config_path}), "
            f"got '{name}'")
    return [str(prompt) for prompt in scenes[name]]


def _validated(context):
    """Everything that can reject an argument, run before any stage starts."""
    segmentation = Node(
        package='perception_suite', executable='3_segmentation_node', output='screen',
        parameters=[{'config': LaunchConfiguration('config'),
                     'prompts': _scene_prompts(context),
                     'scene': LaunchConfiguration('scene'),
                     'backend': LaunchConfiguration('backend')}])
    return [segmentation] + _source(context)


def _source(context):
    name = LaunchConfiguration('source').perform(context)
    if name not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}, got '{name}'")
    executable, arguments = SOURCES[name]
    parameters = {arg: LaunchConfiguration(arg).perform(context) for arg in arguments}
    node = Node(package='perception_suite', executable=executable, name='source_node',
                parameters=[parameters], output='screen')
    # Frames sent while the segmenter is still loading its model are dropped,
    # so the source waits for the stages to come up.
    delay = float(LaunchConfiguration('source_delay').perform(context))
    return [TimerAction(period=delay, actions=[node])]


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('perception_suite'), 'config', 'perception.yaml')
    config = {'config': LaunchConfiguration('config')}

    def stage(executable, extra=None, **kwargs):
        parameters = [config]
        if extra:
            parameters.append(extra)
        return Node(package='perception_suite', executable=executable,
                    parameters=parameters, output='screen', **kwargs)

    return LaunchDescription([
        DeclareLaunchArgument('source', default_value='video',
                              description=f'Input node: {", ".join(sorted(SOURCES))}'),
        DeclareLaunchArgument('source_delay', default_value='10.0',
                              description='Seconds to wait before starting the source'),
        DeclareLaunchArgument('camera_index', default_value='2',
                              description='V4L2 index for source:=camera; the GMSL '
                                          'dashcam enumerates at /dev/video2'),
        DeclareLaunchArgument('camera_fps', default_value='30.0',
                              description='capture rate for source:=camera'),
        DeclareLaunchArgument('video_path', default_value='test_video.mp4',
                              description='Video file for source:=video'),
        DeclareLaunchArgument('loop', default_value='false',
                              description='Restart the video at the end instead of '
                                          'stopping, for source:=video'),
        DeclareLaunchArgument('video_fps', default_value='30.0',
                              description='Publish rate for source:=video; raise it '
                                          'to measure the pipeline ceiling'),
        DeclareLaunchArgument('config', default_value=default_config,
                              description='offroad_autonomy pipeline config YAML'),
        DeclareLaunchArgument('scene', default_value='trail',
                              description='Segmentation prompt set from perception.scenes '
                                          'in the config (e.g. trail, snow, gravel)'),
        DeclareLaunchArgument('backend', default_value='pytorch',
                              choices=['pytorch', 'tensorrt'],
                              description='Segmentation inference: pytorch, or tensorrt '
                                          '(NVIDIA GPU; builds an engine on first use)'),
        DeclareLaunchArgument('speed_mps', default_value='0.0',
                              description='Fixed vehicle speed given to the planner'),
        DeclareLaunchArgument('visualize', default_value='true',
                              description='Open the visualizer window'),
        # First, so a bad source or scene fails before any stage node starts.
        OpaqueFunction(function=_validated),
        stage('2_preprocess_node'),
        stage('4_postprocess_node'),
        stage('5_planner_node', {'speed_mps': LaunchConfiguration('speed_mps')}),
        stage('6_visualizer_node', condition=IfCondition(LaunchConfiguration('visualize'))),
    ])
