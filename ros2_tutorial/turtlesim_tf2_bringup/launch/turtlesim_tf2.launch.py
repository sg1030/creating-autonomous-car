from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():

    # 1) turtlesim node
    turtlesim_node = Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='turtlesim',
        output='screen'
    )

    # 2) Spawn turtle2 after turtlesim is ready
    spawn_turtle2 = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            "ros2 service call /spawn turtlesim/srv/Spawn "
            "\"{x: 4.0, y: 2.0, theta: 0.0, name: 'turtle2'}\""
        ],
        output='screen'
    )

    # 3) TF broadcaster for turtle1
    tf_pub_turtle1 = Node(
        package='turtle_tf2_py',
        executable='turtle_tf2_broadcaster',
        name='tf_pub_turtle1',
        output='screen',
        parameters=[{'turtlename': 'turtle1'}],
        remappings=[('/turtle/pose', '/turtle1/pose')]
    )

    # 4) TF broadcaster for turtle2
    tf_pub_turtle2 = Node(
        package='turtle_tf2_py',
        executable='turtle_tf2_broadcaster',
        name='tf_pub_turtle2',
        output='screen',
        parameters=[{'turtlename': 'turtle2'}],
        remappings=[('/turtle/pose', '/turtle2/pose')]
    )

    # 5) TF listener (makes turtle2 follow turtle1)
    tf_listener = Node(
        package='turtle_tf2_py',
        executable='turtle_tf2_listener',
        name='turtle_tf2_listener',
        output='screen'
    )

    return LaunchDescription([
        turtlesim_node,

        # Wait for turtlesim to start before spawning turtle2
        TimerAction(period=1.0, actions=[spawn_turtle2]),

        # Start TF broadcasters after turtle2 is spawned
        TimerAction(period=2.0, actions=[tf_pub_turtle1, tf_pub_turtle2]),

        # Start TF listener after TF frames are available
        TimerAction(period=3.0, actions=[tf_listener]),
    ])
