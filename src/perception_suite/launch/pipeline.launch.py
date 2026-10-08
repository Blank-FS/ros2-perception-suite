"""Starts the whole pipeline: one source node, then stages 2-6.

    ros2 launch perception_suite pipeline.launch.py
    ros2 launch perception_suite pipeline.launch.py source:=video video_path:=other.mp4
    ros2 launch perception_suite pipeline.launch.py source:=camera
    ros2 launch perception_suite pipeline.launch.py source:=camera camera_index:=0
    ros2 launch perception_suite pipeline.launch.py scene:=snow
    ros2 launch perception_suite pipeline.launch.py visualize:=false
    ros2 launch perception_suite pipeline.launch.py backend:=tensorrt
    ros2 launch perception_suite pipeline.launch.py model:=semantic backend:=tensorrt
    ros2 launch perception_suite pipeline.launch.py config:=demo_route.yaml video_path:=demo_route.mp4

config:= is the input (a YAML file name in src/perception_suite/config),
model:= the segmentation model, backend:= how it runs and scene:= its prompts;
see perception_suite/settings.py. Run it from the workspace root, where the
relative video and weight paths point.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from perception_suite import settings

# Source name -> (executable, launch arguments it takes as parameters).
# A new input (camera, beamng) is one entry here plus its node.
SOURCES = {
    'video': ('1_video_node', ['video_path', 'loop', 'video_fps', 'width']),
    'camera': ('1_camera_node', ['camera_index', 'camera_fps', 'width']),
}


def _validated(context):
    """Everything that can reject an argument, run before any stage starts."""
    path = settings.resolve(LaunchConfiguration('config').perform(context))
    settings.model(path, LaunchConfiguration('model').perform(context))
    settings.scene_prompts(path, LaunchConfiguration('scene').perform(context))
    parameters = {'config': LaunchConfiguration('config'),
                  'model': LaunchConfiguration('model'),
                  'scene': LaunchConfiguration('scene'),
                  'backend': LaunchConfiguration('backend')}
    segmentation = Node(
        package='perception_suite', executable='3_segmentation_node', output='screen',
        parameters=[parameters])
    return [segmentation] + _source(context)


def _source(context):
    name = LaunchConfiguration('source').perform(context)
    if name not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}, got '{name}'")
    executable, arguments = SOURCES[name]
    parameters = {arg: LaunchConfiguration(arg).perform(context) for arg in arguments}
    # perform() yields a string and both source nodes declare `width` with an
    # integer default, which rclpy rejects with InvalidParameterTypeException.
    parameters['width'] = int(parameters['width'])
    node = Node(package='perception_suite', executable=executable, name='source_node',
                parameters=[parameters], output='screen')
    # Frames sent while the segmenter is still loading its model are dropped,
    # so the source waits for the stages to come up.
    delay = float(LaunchConfiguration('source_delay').perform(context))
    return [TimerAction(period=delay, actions=[node])]


def generate_launch_description():
    config = {'config': LaunchConfiguration('config'), 'model': LaunchConfiguration('model')}

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
        DeclareLaunchArgument('width', default_value='1280',
                              description='Width the source publishes. Nothing between '
                                          'the source and 2_preprocess_node reads the '
                                          'larger frame, so setting this to the working '
                                          'width removes a resample and 4x the bytes'),
        DeclareLaunchArgument('window_x', default_value='-1',
                              description='Dashboard window position; -1 leaves it to '
                                          'the window manager'),
        DeclareLaunchArgument('window_y', default_value='-1'),
        DeclareLaunchArgument('video_path', default_value='test_video.mp4',
                              description='Video file for source:=video'),
        DeclareLaunchArgument('loop', default_value='true',
                              description='Restart the video at the end instead of '
                                          'stopping, for source:=video; false plays it once'),
        DeclareLaunchArgument('video_fps', default_value='30.0',
                              description='Publish rate for source:=video; raise it '
                                          'to measure the pipeline ceiling'),
        DeclareLaunchArgument('config', default_value=settings.DEFAULT_CONFIG,
                              description='Pipeline config for the input: a YAML file '
                                          'name in src/perception_suite/config'),
        DeclareLaunchArgument('model', default_value=settings.DEFAULT_MODEL,
                              description='Segmentation model from perception.models in '
                                          'the config: yoloe or semantic'),
        DeclareLaunchArgument('scene', default_value=settings.DEFAULT_SCENE,
                              description='YOLOE prompt set from perception.scenes in the '
                                          'config (e.g. trail, snow, sandy); ignored by '
                                          'the semantic model'),
        DeclareLaunchArgument('backend', default_value='pytorch',
                              choices=['pytorch', 'tensorrt'],
                              description='Segmentation inference: pytorch, or tensorrt '
                                          '(NVIDIA GPU; builds an engine on first use)'),
        DeclareLaunchArgument('speed_mps', default_value='0.0',
                              description='Fixed vehicle speed given to the planner'),
        DeclareLaunchArgument('visualize', default_value='true',
                              description='Open the visualizer window'),
        # First, so a bad config, model, scene or source fails before any
        # stage node starts.
        OpaqueFunction(function=_validated),
        stage('2_preprocess_node'),
        stage('4_postprocess_node'),
        stage('5_planner_node', {'speed_mps': LaunchConfiguration('speed_mps')}),
        stage('6_visualizer_node',
              {'window_x': LaunchConfiguration('window_x'),
               'window_y': LaunchConfiguration('window_y')},
              condition=IfCondition(LaunchConfiguration('visualize'))),
    ])
