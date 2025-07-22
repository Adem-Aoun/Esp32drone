import math, rclpy, os, time
from math import ceil, nan, isfinite
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.duration import Duration
from px4_msgs.msg import (VehicleCommand, OffboardControlMode,
                          TrajectorySetpoint, VehicleLocalPosition, VehicleStatus,
                          BatteryStatus, EstimatorStatusFlags, FailsafeFlags)
from nav_msgs.msg import Path
from geometry_msgs.msg import (PoseStamped, TransformStamped, Point, 
                               Vector3, Quaternion, PoseArray, Pose)
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Range, CompressedImage
from cv_bridge import CvBridge
import cv2
import threading
import queue
import numpy as np

# Import the action definition
from scan_mission_interfaces.action import FenceMission

# Camera constants
DEFAULT_SENSOR_WIDTH = 3.68e-3  # 3.68 mm (1/4" sensor effective width)
DEFAULT_FOCAL_LENGTH = 3.0e-3   # 3.0 mm (typical focal length)

def calculate_default_hfov():
    hfov_rad = 2 * math.atan(DEFAULT_SENSOR_WIDTH / (2 * DEFAULT_FOCAL_LENGTH))
    return math.degrees(hfov_rad)

DEFAULT_HFOV = calculate_default_hfov()  # ~62.8°

class VideoRecorder:
    def __init__(self, node):
        self.node = node
        self.recording = False
        self.writer = None
        self.image_queue = queue.Queue(maxsize=30)  # Buffer 1 second at 30fps
        self.thread = None
        self.bridge = CvBridge()
        self.output_dir = "mission_recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get video parameters from node
        self.width = node.get_parameter('video_width').value if node.has_parameter('video_width') else 640
        self.height = node.get_parameter('video_height').value if node.has_parameter('video_height') else 480
        self.fps = node.get_parameter('video_fps').value if node.has_parameter('video_fps') else 30
        self.topic = "/image_raw"
        
        # Subscribe to image topic
        self.sub = self.node.create_subscription(
            CompressedImage,
            self.topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        self.node.get_logger().info(f"Video recorder initialized: {self.width}x{self.height} @ {self.fps}fps")

    def start_recording(self):
        if self.recording:
            self.node.get_logger().warning("Recording already in progress")
            return False
            
        try:
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.filename = os.path.join(self.output_dir, f"mission_{timestamp}.mp4")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.filename,
                fourcc,
                self.fps,
                (self.width, self.height))
            
            if not self.writer.isOpened():
                self.node.get_logger().error("Failed to open video writer")
                return False
                
            # Clear the queue
            while not self.image_queue.empty():
                self.image_queue.get_nowait()
                
            # Start processing thread
            self.recording = True
            self.thread = threading.Thread(target=self.process_frames)
            self.thread.start()
            
            self.node.get_logger().info(f"Started video recording: {self.filename}")
            return True
        except Exception as e:
            self.node.get_logger().error(f"Failed to start recording: {e}")
            return False

    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        # Release writer
        if self.writer:
            self.writer.release()
            self.writer = None
            self.node.get_logger().info(f"Video recording saved: {self.filename}")

    def image_callback(self, msg):
        if not self.recording:
            return
            
        try:
            # Put compressed image in queue without processing
            self.image_queue.put_nowait(msg)
        except queue.Full:
            # Drop frame if queue is full
            pass

    def process_frames(self):
        while self.recording or not self.image_queue.empty():
            try:
                # Get next image with timeout
                msg = self.image_queue.get(timeout=0.5)
                
                # Convert to OpenCV
                cv_img = self.bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding='bgr8')
                
                # Check if image is valid
                if cv_img is None or cv_img.size == 0:
                    self.node.get_logger().warn("Received empty image frame")
                    continue
                
                # Resize if needed
                h, w = cv_img.shape[:2]
                if (w, h) != (self.width, self.height):
                    cv_img = cv2.resize(cv_img, (self.width, self.height))
                
                # Write frame
                if self.writer:
                    self.writer.write(cv_img)
            except queue.Empty:
                continue
            except Exception as e:
                self.node.get_logger().error(f"Video processing error: {e}")

class FenceMissionActionServer(Node):
    def __init__(self):
        super().__init__("fence_mission_action_server")
        self.get_logger().info("╔════════════════════════════════════════╗")
        self.get_logger().info("║  ROBUST FENCE MISSION ACTION SERVER    ║")
        self.get_logger().info("╚════════════════════════════════════════╝")
        
        # Declare parameters for video recording
        self.declare_parameter('video_width', 640)
        self.declare_parameter('video_height', 480)
        self.declare_parameter('video_fps', 30)
        
        # ─── Action Server ─────────────────────────────────
        self._action_server = ActionServer(
            self,
            FenceMission,
            'fence_mission',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.current_goal_handle = None
        self.mission_active = False
        self.mission_cancelled = False
        self.rtl_cmd_sent = False
        self.rtl_state_logged = False 
        self.rtl_start_time = None
        
        # ─── DDS QoS SETTINGS ──────────────────────────────
        self.get_logger().info("Configuring QoS profiles...")
        
        # For subscribers listening to PX4 (using sensor data profile)
        self.sub_qos = qos_profile_sensor_data
        
        # For publishers sending to PX4 (using transient local for commands)
        self.pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.get_logger().info(f"Subscriber QoS: {self.sub_qos}")
        self.get_logger().info(f"Publisher QoS: {self.pub_qos}")

        # ─── DDS pubs / subs ───────────────────────────────
        self.get_logger().info("Initializing publishers...")
        self.ctrl_pub = self.create_publisher(OffboardControlMode,
                                              "/fmu/in/offboard_control_mode", 
                                              self.pub_qos)
        self.sp_pub   = self.create_publisher(TrajectorySetpoint,
                                              "/fmu/in/trajectory_setpoint",  
                                              self.pub_qos)
        self.cmd_pub  = self.create_publisher(VehicleCommand,
                                              "/fmu/in/vehicle_command",      
                                              self.pub_qos)
                                              
        # ROS topics use default QoS
        self.path_pub = self.create_publisher(Path, "/scan_path", 10)
        self.geofence_pub = self.create_publisher(Marker, "/geofence", 10)
        self.obstacle_marker_pub = self.create_publisher(MarkerArray, "/obstacle_markers", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Static TF broadcaster for camera
        self.camera_tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_camera_tf()
        
        self.get_logger().info("Initializing subscribers...")
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position",
                                 self.cb_pos, self.sub_qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status",
                                 self.cb_status, self.sub_qos)
        self.create_subscription(BatteryStatus, "/fmu/out/battery_status",
                                 self.cb_battery, self.sub_qos)
        self.create_subscription(EstimatorStatusFlags, "/fmu/out/estimator_status_flags",
                                 self.cb_estimator_flags, self.sub_qos)
        self.create_subscription(FailsafeFlags, "/fmu/out/failsafe_flags",
                                 self.cb_failsafe_flags, self.sub_qos)
        
        # Ultrasonic sensors use sensor data profile
        self.create_subscription(Range, '/ultrasonic_sensor/forward/filtered', 
                                 self.cb_ultrasonic_front, self.sub_qos)
        self.create_subscription(Range, '/ultrasonic_sensor/back/filtered', 
                                 self.cb_ultrasonic_back, self.sub_qos)
        self.create_subscription(Range, '/ultrasonic_sensor/left/filtered', 
                                 self.cb_ultrasonic_left, self.sub_qos)
        self.create_subscription(Range, '/ultrasonic_sensor/right/filtered', 
                                 self.cb_ultrasonic_right, self.sub_qos)
        self.create_subscription(Range, '/ultrasonic_sensor/downward/filtered', 
                                 self.cb_ultrasonic_down, self.sub_qos)

        # ─── Video Recorder ───────────────────────────────
        self.video_recorder = VideoRecorder(self)
        self.recording_active = False

        # ─── Controller Parameters ────────────────────────
        self.get_logger().info("Configuring controller parameters...")
        # Base controller gains (conservative)
        self.kp_x_base, self.vx_max_base = 0.6, 0.80
        self.kp_y_base, self.vy_max_base = 0.6, 0.50
        self.speed_factor = 1.0  # Speed multiplier
        
        # Velocity limits
        self.vx_max = self.vx_max_base
        self.vy_max = self.vy_max_base
        self.decel_dist = 1.0
        self.tol_x = self.tol_y = 0.30
        self.tol_z = 0.10
        self.track_tol = 0.20
        self.hover_secs = 1.0
        self.arming_sent = False
        
        # Integral term for standard controller
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.ki = 0.05  # Integral gain
        self.integral_max = 0.5  # Anti-windup limit
        
        # ─── mission state ────────────────────────────────
        self.waypoints = []
        self.wp_idx = 0
        self.pos  = VehicleLocalPosition()
        self.stat = VehicleStatus()
        self.battery = BatteryStatus()
        self.estimator_flags = EstimatorStatusFlags()
        self.failsafe_flags = FailsafeFlags()
        self.armed = False
        self.last_position = (0.0, 0.0, 0.0)  # (x, y, z)

        self.phase = "IDLE"
        self.previous_phase = None
        self.hover_t0 = None
        self.sp_warm  = 0
        self.loop_counter = 0
        self.safety_reason = None
        self.last_safety_log = 0
        self.low_battery_warning_sent = False

        # ─── Path tracking ────────────────────────────────
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        
        # ─── fence-recovery bookkeeping ───────────────────
        self.breach = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.saved_wp = 0
        self.inside_fence = True

        # ─── Obstacle avoidance ───────────────────────────
        self.obstacle_detected = {
            'front': False,
            'back': False,
            'left': False,
            'right': False,
            'down': False
        }
        self.ultrasonic_readings = {
            'front': float('inf'),
            'back': float('inf'),
            'left': float('inf'),
            'right': float('inf'),
            'down': float('inf')
        }
        self.obstacle_positions = []
        self.avoid_direction = None
        self.escape_velocity = 0.5
        self.obstacle_threshold = 2.0

        self.get_logger().info("Starting control timer...")
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("╔════════════════════════════════════════╗")
        self.get_logger().info("║  ACTION SERVER READY                   ║")
        self.get_logger().info("╚════════════════════════════════════════╝")
        self.get_logger().info(f"Camera default HFOV: {DEFAULT_HFOV:.1f}°")

    # ─── Camera TF Publisher ──────────────────────────────
    def publish_camera_tf(self):
        """Publish static transform for camera position"""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "camera_link"
        
        # Position: 10cm in front (+X) and 10cm below (-Z) center
        transform.transform.translation.x = 0.10
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = -0.10
        
        # Rotation: -90° pitch (pointing downward)
        q = quaternion_from_euler(0, -math.pi/2, 0)
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        
        self.camera_tf_broadcaster.sendTransform(transform)
        self.get_logger().info("Published camera static transform")

    # ─── Action Server Callbacks ──────────────────────────
    def goal_callback(self, goal_request):
        """Accept or reject new goals"""
        if self.mission_active:
            self.get_logger().warn("Mission already in progress - rejecting new goal")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation requests"""
        self.get_logger().info("Received cancellation request")
        self.mission_cancelled = True
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the mission when a new goal is accepted"""
        # Set process priority for better real-time performance
        try:
            import os
            os.nice(-10)
            self.get_logger().info("Set process priority to high")
        except:
            self.get_logger().warn("Could not set process priority")
            
        self.current_goal_handle = goal_handle
        goal = goal_handle.request
        self.rtl_cmd_sent = False  # Reset for new missions
        self.low_battery_warning_sent = False
        
        # Initialize mission parameters from action goal
        L = goal.scan_length
        W = goal.scan_width
        altitude = goal.target_altitude
        self.enable_safety_actions = goal.fail_safe
        self.enable_geofence = goal.geofence_enabled
        self.enable_obstacle_avoidance = goal.obstacle_avoidance
        
        # Reset integral terms
        self.integral_x = 0.0
        self.integral_y = 0.0
        
        # Handle speed parameter
        if goal.speed_scan > 0:
            self.speed_factor = goal.speed_scan / self.vx_max_base
            self.get_logger().info(f"Speed factor set to: {self.speed_factor:.2f}")
        else:
            self.speed_factor = 1.0
            self.get_logger().info("Using default speed")

        # Handle FOV parameter - use default if not provided
        if goal.fov <= 0:
            HFOV_deg = DEFAULT_HFOV
            self.get_logger().info(f"Using default HFOV: {HFOV_deg:.1f}°")
        else:
            HFOV_deg = goal.fov
            self.get_logger().info(f"Using custom HFOV: {HFOV_deg:.1f}°")

        # Log mission parameters
        status_safety = "ENABLED" if self.enable_safety_actions else "DISABLED"
        status_geofence = "ENABLED" if self.enable_geofence else "DISABLED"
        self.get_logger().info(f"New mission received: L={L}m, W={W}m, alt={altitude}m")
        self.get_logger().info(f"Safety actions: {status_safety}")
        self.get_logger().info(f"Geofence: {status_geofence}")

        # Compute mission parameters
        HFOV_rad = math.radians(HFOV_deg)
        overlap = 0.1
        fov_width = 2 * altitude * math.tan(HFOV_rad / 2)
        self.half_fov_offset = -fov_width / 2.0
        step = fov_width * (1 - overlap)
        self.decel_dist = 1.0
        self.min_speed = 0.1*self.vx_max*self.speed_factor
        self.step_x = step
        self.target_altitude = altitude
        self.z_ref = -altitude

        self.length_y = L
        self.width_x  = W
        self.n_rows   = max(1, ceil(W / step))
        
        # Geofence parameters (only if enabled)
        if self.enable_geofence:
            margin = 1.0
            self.fence_min_x = -self.width_x - margin
            self.fence_max_x = margin
            self.fence_min_y = -margin
            self.fence_max_y = self.length_y + margin
            self.get_logger().info(f"Geofence active: X[{self.fence_min_x:.1f}, {self.fence_max_x:.1f}], Y[{self.fence_min_y:.1f}, {self.fence_max_y:.1f}]")
        else:
            self.get_logger().info("Geofence disabled")

        # Generate waypoints with approach/departure paths
        self.waypoints = self._make_path()
        approach = [(self.half_fov_offset, -1.0)]  # Approach path
        departure = [(self.waypoints[-1][0], self.waypoints[-1][1] + 1.0)]  # Departure path
        self.waypoints = approach + self.waypoints + departure
        
        self.get_logger().info(f"Generated {len(self.waypoints)} waypoints")
        self.get_logger().info(f"Step size: {step:.2f}m, FOV width: {fov_width:.2f}m")

        # Reset mission state
        self.wp_idx = 0
        self.phase = "TAKEOFF"
        self.previous_phase = None
        self.hover_t0 = None
        self.safety_reason = None
        self.obstacle_positions = []
        self.path_msg.poses = []
        self.mission_active = True
        self.mission_cancelled = False
        
        # Start video recording if requested
        self.recording_active = goal.video_recording
        if self.recording_active:
            if self.video_recorder.start_recording():
                self.get_logger().info("Video recording started")
            else:
                self.get_logger().error("Failed to start video recording")
                self.recording_active = False
        else:
            self.get_logger().info("Video recording not requested")
        
        # Publish initial feedback
        feedback = FenceMission.Feedback()
        feedback.status_update = "Mission initialized"
        goal_handle.publish_feedback(feedback)

        # Mission execution loop
        while rclpy.ok() and self.mission_active:
            # Check if mission was cancelled
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Mission cancellation received")
                goal_handle.canceled()
                self.mission_active = False
                self.phase = "RTL"
                self.rtl()
                break
                
            rclpy.spin_once(self, timeout_sec=0.1)

        # Stop video recording if active
        if self.recording_active:
            self.video_recorder.stop_recording()
            self.recording_active = False

        # Return result when mission completes
        result = FenceMission.Result()
        result.success = (self.phase == "RTL_COMPLETE")
        result.log_file = "mission_log.txt"
        
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
            
        return result

    # ─── path generator ───────────────────────────────────
    def _make_path(self):
        self.get_logger().info("Building lawn-mower pattern...")
        pts, x = [], self.half_fov_offset
        for r in range(self.n_rows):
            y_end = self.length_y if r % 2 == 0 else 0.0
            pts.append((x, y_end))          # row end
            if r < self.n_rows - 1:
                x -= self.step_x            # shift −X
                pts.append((x, y_end))      # after shift
        return pts

    # ─── PX4 helper wrappers ──────────────────────────────
    def _now_us(self): return int(self.get_clock().now().nanoseconds / 1e3)
    def _clamp(self, v, l): return max(-l, min(v, l))

    def _cmd(self, cmd, **p):
        m = VehicleCommand()
        m.command, m.from_external = cmd, True
        m.target_system = m.source_system = 1
        m.target_component = m.source_component = 1
        for i in range(1, 8):
            setattr(m, f"param{i}", float(p.get(f"param{i}", 0.0)))
        m.timestamp = self._now_us()
        self.cmd_pub.publish(m)
        self.get_logger().debug(f"Published command: {cmd} with params {p}")

    def arm(self):
        self.get_logger().info("Sending ARM command...")
        self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        
    def offboard(self):
        self.get_logger().info("Requesting OFFBOARD mode...")
        self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        
    def land(self):
        self.get_logger().info("⏬⏬⏬ LANDING COMMAND SENT ⏬⏬⏬")
        self._cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        
    def rtl(self):
        self.get_logger().warning("🛟🛟🛟 RETURN TO LAUNCH COMMAND SENT 🛟🛟🛟")
        self._cmd(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)

    # Always enable position control for altitude management
    def _ctrl_mode(self):
        hb = OffboardControlMode()
        hb.position = True
        hb.velocity = True
        hb.timestamp = self._now_us()
        self.ctrl_pub.publish(hb)

    # Mixed setpoint helper for PX4-managed altitude
    def _sp_mixed(self, vx=nan, vy=nan, vz=nan, z=nan):
        sp = TrajectorySetpoint()
        sp.velocity = [float(vx), float(vy), float(vz)]
        sp.position = [nan, nan, float(z)]
        sp.yaw = 1.5708
        sp.timestamp = self._now_us()
        self.sp_pub.publish(sp)

    # Position setpoint for takeoff
    def _sp_pos(self, x, y, z):
        sp = TrajectorySetpoint()
        sp.position = [float(x), float(y), float(z)]
        sp.velocity = [nan, nan, nan]
        sp.yaw = 1.5708
        sp.timestamp = self._now_us()
        self.sp_pub.publish(sp)

    # ─── hover helpers ───────────────────────────────────
    def _hover_hold(self):
        self._sp_mixed(0.0, 0.0, nan, self.z_ref)

    def _hover_done(self):
        elapsed = (self.get_clock().now() - self.hover_t0).nanoseconds / 1e9
        return elapsed >= self.hover_secs

    # ─── callbacks ────────────────────────────────────────
    def cb_pos(self, m): 
        self.pos = m
        if self.mission_active and self.pos.xy_valid and self.pos.z_valid:
            self.publish_path_and_tf()

    def cb_status(self, m):
        self.stat = m
        new_armed = (m.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        
        if new_armed != self.armed:
            self.armed = new_armed
            self.get_logger().info(f"ARM STATUS CHANGED: {'ARMED' if new_armed else 'DISARMED'}")
        else:
            self.armed = new_armed
            
        # Track RTL state
        if self.phase == "RTL" and not self.rtl_state_logged:
            if m.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_RTL:
                self.get_logger().info("RTL in progress...")
                self.rtl_state_logged = True
                
    def cb_battery(self, m):
        self.battery = m
        if self.mission_active and self.loop_counter % 20 == 0:
            # Calculate actual state of charge percentage
            if m.capacity > 0:
                soc_percent = (m.remaining / m.capacity) * 100.0
            else:
                soc_percent = 0.0
            
            # Enhanced battery monitoring
            voltage_per_cell = m.voltage_v / m.cell_count if m.cell_count > 0 else 0
            
            status = "NONE"
            if m.warning == BatteryStatus.WARNING_LOW:
                status = "LOW"
            elif m.warning == BatteryStatus.WARNING_CRITICAL:
                status = "CRITICAL"
            elif m.warning == BatteryStatus.WARNING_EMERGENCY:
                status = "EMERGENCY"
                
            self.get_logger().info(
                f"BATTERY: {soc_percent:.1f}% | "
                f"Voltage: {m.voltage_v:.1f}V ({voltage_per_cell:.2f}V/cell) | "
                f"Status: {status}"
            )
            
    def cb_estimator_flags(self, m):
        self.estimator_flags = m
            
    def cb_failsafe_flags(self, m):
        self.failsafe_flags = m

    # Ultrasonic sensor callbacks with NaN/inf handling
    def cb_ultrasonic_front(self, msg):
        if not isfinite(msg.range):
            self.ultrasonic_readings['front'] = float('inf')
            self.obstacle_detected['front'] = False
        else:
            self.ultrasonic_readings['front'] = msg.range
            self.obstacle_detected['front'] = (msg.range <= self.obstacle_threshold)
        
    def cb_ultrasonic_back(self, msg):
        if not isfinite(msg.range):
            self.ultrasonic_readings['back'] = float('inf')
            self.obstacle_detected['back'] = False
        else:
            self.ultrasonic_readings['back'] = msg.range
            self.obstacle_detected['back'] = (msg.range <= self.obstacle_threshold)
        
    def cb_ultrasonic_left(self, msg):
        if not isfinite(msg.range):
            self.ultrasonic_readings['left'] = float('inf')
            self.obstacle_detected['left'] = False
        else:
            self.ultrasonic_readings['left'] = msg.range
            self.obstacle_detected['left'] = (msg.range <= self.obstacle_threshold)
        
    def cb_ultrasonic_right(self, msg):
        if not isfinite(msg.range):
            self.ultrasonic_readings['right'] = float('inf')
            self.obstacle_detected['right'] = False
        else:
            self.ultrasonic_readings['right'] = msg.range
            self.obstacle_detected['right'] = (msg.range <= self.obstacle_threshold)
        
    def cb_ultrasonic_down(self, msg):
        if not isfinite(msg.range):
            self.ultrasonic_readings['down'] = float('inf')
            self.obstacle_detected['down'] = False
        else:
            self.ultrasonic_readings['down'] = msg.range
            self.obstacle_detected['down'] = (msg.range <= self.obstacle_threshold)

    # ─── Publish path and TF ──────────────────────────────
    def publish_path_and_tf(self):
        # Convert PX4 NED to ROS ENU
        enu_x = self.pos.y
        enu_y = self.pos.x
        enu_z = -self.pos.z
        
        # Create new pose stamped
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = enu_x
        pose.pose.position.y = enu_y
        pose.pose.position.z = enu_z
        
        # Use heading if available
        if hasattr(self.pos, 'heading_good_for_control') and self.pos.heading_good_for_control:
            yaw_enu = math.pi/2 - self.pos.heading
            q = quaternion_from_euler(0, 0, yaw_enu)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            orientation = q
        else:
            pose.pose.orientation.w = 1.0
            orientation = [0.0, 0.0, 0.0, 1.0]
        
        # Append to path and publish
        self.path_msg.poses.append(pose)
        self.path_msg.header.stamp = pose.header.stamp
        self.path_pub.publish(self.path_msg)
        
        # Create and broadcast transform
        transform = TransformStamped()
        transform.header.stamp = pose.header.stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = enu_x
        transform.transform.translation.y = enu_y
        transform.transform.translation.z = enu_z
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        
        self.tf_broadcaster.sendTransform(transform)

    # ─── Geofence marker publisher ────────────────────────
    def publish_geofence_marker(self):
        if not self.mission_active or not self.enable_geofence:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "geofence"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        
        if self.inside_fence:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
        marker.pose.orientation.w = 1.0
        
        points = [
            (self.fence_min_x, self.fence_min_y),
            (self.fence_max_x, self.fence_min_y),
            (self.fence_max_x, self.fence_max_y),
            (self.fence_min_x, self.fence_max_y),
            (self.fence_min_x, self.fence_min_y)
        ]
        
        for x_ned, y_ned in points:
            p = Point()
            p.x = y_ned
            p.y = x_ned
            p.z = 0.0
            marker.points.append(p)
            
        marker.lifetime = Duration().to_msg()
        self.geofence_pub.publish(marker)

    # ─── Obstacle marker publisher ────────────────────────
    def publish_obstacle_markers(self):
        if not self.mission_active or not self.obstacle_positions:
            return
            
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, (x_ned, y_ned, z_ned) in enumerate(self.obstacle_positions):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = y_ned
            marker.pose.position.y = x_ned
            marker.pose.position.z = -z_ned
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.lifetime = Duration(seconds=30).to_msg()
            marker_array.markers.append(marker)
        
        self.obstacle_marker_pub.publish(marker_array)

    # ─── fence helpers ───────────────────────────────────
    def _inside(self, x, y):
        return (self.fence_min_x <= x <= self.fence_max_x and
                self.fence_min_y <= y <= self.fence_max_y)

    def _nearest_on_fence(self, x, y):
        return (min(max(x, self.fence_min_x), self.fence_max_x),
                min(max(y, self.fence_min_y), self.fence_max_y))

    # ─── Obstacle detection helpers ──────────────────────
    def any_obstacle_detected(self):
        return any(self.obstacle_detected.values())
    
    def all_clear(self):
        return all(reading > self.obstacle_threshold for reading in self.ultrasonic_readings.values())
    
    def determine_escape_direction(self):
        opposites = {
            'front': 'back',
            'back': 'front',
            'left': 'right',
            'right': 'left',
            'down': 'up'
        }
        
        blocked_dirs = [dir for dir in ['front','back','left','right','down'] 
                        if self.obstacle_detected[dir]]
        
        if 'down' in blocked_dirs:
            return 'up'
        
        for blocked_dir in blocked_dirs:
            opp = opposites[blocked_dir]
            if not self.obstacle_detected.get(opp, False):
                return opp
        
        return 'up'

    def check_safety(self):
        # Enhanced battery monitoring
        if self.battery.capacity > 0:
            soc_percent = (self.battery.remaining / self.battery.capacity) * 100.0
            voltage_per_cell = self.battery.voltage_v / self.battery.cell_count
        else:
            soc_percent = 0.0
            voltage_per_cell = 0.0
        
        # Battery safety checks
        if soc_percent < 10.0 or voltage_per_cell < 3.5:
            return "CRITICAL_BATTERY", "RTL"
        elif soc_percent < 20.0 or voltage_per_cell < 3.6:
            return "LOW_BATTERY", "RTL"
        
        # Existing non-battery safety checks
        if not self.enable_safety_actions:
            return None, None

        if (self.failsafe_flags.fd_critical_failure or 
            self.failsafe_flags.fd_motor_failure or
            self.failsafe_flags.battery_unhealthy):
            return "CRITICAL_SYSTEM_FAILURE", "LAND"
            
        if (not self.pos.xy_valid or not self.pos.z_valid or
            self.failsafe_flags.local_position_invalid):
            return "ESTIMATION_FAILURE", "LAND"
            
        if self.failsafe_flags.wind_limit_exceeded:
            return "HIGH_WIND", "RTL"
            
        if self.failsafe_flags.mission_failure:
            return "MISSION_FAILURE", "RTL"
            
        return None, None

    # ─── main control loop ───────────────────────────────
    def control_loop(self):
        self.loop_counter += 1
        
        # Only run mission logic when active
        if not self.mission_active:
            return
            
        self._ctrl_mode()
        x, y, z = self.pos.x, self.pos.y, self.pos.z

        # Publish visualizations periodically
        if self.loop_counter % 10 == 0:
            if self.pos.xy_valid and self.pos.z_valid:
                self.publish_path_and_tf()
            if self.enable_geofence:
                self.publish_geofence_marker()
            if self.obstacle_positions:
                self.publish_obstacle_markers()

        # Publish feedback
        if self.loop_counter % 10 == 0 and self.current_goal_handle:
            feedback = FenceMission.Feedback()
            
            # Improved feedback messages
            if self.phase == "TAKEOFF":
                if self.pos.z_valid:
                    current_alt = -self.pos.z
                    target_alt = -self.z_ref
                    feedback.status_update = f"TAKEOFF: {current_alt:.1f}/{target_alt:.1f}m"
                else:
                    feedback.status_update = "TAKEOFF: Ascending"
            elif self.phase == "OFFSET":
                feedback.status_update = "OFFSET: Moving to start position"
            elif self.phase == "HOVER_START":
                elapsed = (self.get_clock().now() - self.hover_t0).nanoseconds / 1e9
                remaining = max(0, self.hover_secs - elapsed)
                feedback.status_update = f"HOVER: {remaining:.1f}s remaining"
            elif self.phase == "MISSION":
                progress = (self.wp_idx + 1) / len(self.waypoints) * 100
                feedback.status_update = f"SCANNING: WP {self.wp_idx+1}/{len(self.waypoints)} ({progress:.0f}%)"
            else:
                feedback.status_update = f"{self.phase}"
            
            self.current_goal_handle.publish_feedback(feedback)

        # Safety checks
        if self.enable_safety_actions:
            reason, action = self.check_safety()
            if reason and action and self.phase not in ["SAFETY", "LAND"]:
                self.safety_reason = reason
                self.get_logger().error(f"SAFETY: {reason} - Triggering {action}")
                self.phase = "SAFETY"
                
                if action == "LAND":
                    self.land()
                elif action == "RTL":
                    self.rtl()
                    
            if self.safety_reason and self.loop_counter % 10 == 0:
                self.get_logger().error(f"SAFETY ACTIVE: {self.safety_reason}")

        # Check fence status (only if enabled)
        if self.enable_geofence and self.phase != "AVOID":
            current_inside = self._inside(x, y)
            if current_inside != self.inside_fence:
                self.inside_fence = current_inside
                status = "INSIDE" if self.inside_fence else "BREACH"
                self.get_logger().warning(f"FENCE STATUS: {status}")

        # Fence breach detection (only if enabled)
        if (self.enable_geofence and
            self.phase not in {"LAND", "TAKEOFF", "AVOID", "SAFETY"} and 
            not self.inside_fence):
            self.breach  = (x, y)
            self.target  = self._nearest_on_fence(x, y)
            self.saved_wp = self.wp_idx
            self.phase = "FENCE_OUT"
            self.get_logger().error(f"GEOFENCE BREACH DETECTED!")
            return

        # Obstacle detection
        if (self.phase in {"MISSION", "HOVER_WP"} and 
            self.any_obstacle_detected()):
            self.previous_phase = self.phase
            self.phase = "AVOID"
            self.avoid_direction = None
            self.get_logger().warning("OBSTACLE DETECTED!")

        # FSM logic
        if self.phase == "TAKEOFF":
            if not self.armed:
                if not self.arming_sent:
                    self.get_logger().info("Attempting to ARM and enable OFFBOARD")
                    self.arm()
                    self.offboard()
                    self.arming_sent = True
            else:
                # Drone is armed - proceed with takeoff
                self._sp_mixed(0.0, 0.0, -0.5, self.z_ref)  # Gentle ascent
                
                # Log ascent progress
                if self.loop_counter % 10 == 0 and self.pos.z_valid:
                    current_alt = -self.pos.z
                    target_alt = -self.z_ref
                    self.get_logger().info(
                        f"Ascending: {current_alt:.1f}/{target_alt:.1f}m "
                        f"({abs(self.pos.z - self.z_ref):.1f}m to go)"
                    )
                
                # Check if target altitude reached
                if abs(self.pos.z - self.z_ref) < self.tol_z:
                    self.get_logger().info("✅ TAKEOFF COMPLETE")
                    self.phase = "OFFSET"
                    self.get_logger().info(
                        f"✅ TAKEOFF COMPLETE — sliding +X by {self.half_fov_offset:.2f} m"
                    )
        elif self.phase == "OFFSET":
            dx   = self.half_fov_offset - self.pos.x
            dist = abs(dx)

            # linear ramp from full speed → min_speed
            if dist < self.decel_dist:
                speed = max(self.min_speed,
                            self.vx_max * self.speed_factor * (dist / self.decel_dist))
            else:
                speed = self.vx_max * self.speed_factor

            vx = math.copysign(speed, dx)
            self._sp_mixed(vx, 0.0, nan, self.z_ref)

            # arrival?
            if dist < self.tol_x:
                self.get_logger().info("✅ OFFSET COMPLETE")
                self.phase   = "HOVER_START"
                self.hover_t0 = self.get_clock().now()

        elif self.phase == "HOVER_START":
            self._hover_hold()
            if self._hover_done():
                self.phase = "MISSION"
                self.get_logger().info("MISSION START")
        elif self.phase == "MISSION":
            tx, ty = self.waypoints[self.wp_idx]
            dx, dy = tx - x, ty - y
            dist   = math.hypot(dx, dy)
            
            # Add integral term to reduce steady-state error
            dt = 0.1  # Control loop period
            self.integral_x += dx * dt
            self.integral_y += dy * dt
            
            # Anti-windup clamping
            self.integral_x = max(-self.integral_max, min(self.integral_x, self.integral_max))
            self.integral_y = max(-self.integral_max, min(self.integral_y, self.integral_max))
            
            # Calculate velocity demands with integral term
            raw_vx = self.kp_x_base * dx + self.ki * self.integral_x
            raw_vy = self.kp_y_base * dy + self.ki * self.integral_y

            # ramp from full speed → min_speed
            if dist < self.decel_dist:
                scale_v = max(self.min_speed / (self.vx_max * self.speed_factor),
                              dist / self.decel_dist)
            else:
                scale_v = 1.0

            vx = self._clamp(raw_vx, self.vx_max * self.speed_factor) * scale_v
            vy = self._clamp(raw_vy, self.vy_max * self.speed_factor) * scale_v

            self._sp_mixed(vx, vy, nan, self.z_ref)

            # arrived?
            if dist < self.track_tol + 0.1:
                self.get_logger().info(f"WAYPOINT {self.wp_idx} REACHED")
                # Reset integral terms when reaching waypoint
                self.integral_x = 0.0
                self.integral_y = 0.0
                
                # if there's another WP, just advance—no hover
                if self.wp_idx < len(self.waypoints) - 1:
                    self.wp_idx += 1
                else:
                    # last waypoint → hover then RTL
                    self.phase   = "HOVER_WP"
                    self.hover_t0 = self.get_clock().now()

        elif self.phase == "HOVER_WP":
            self._hover_hold()
            if self._hover_done():
                self.wp_idx += 1
                if self.wp_idx < len(self.waypoints):
                    self.phase = "MISSION"
                    self.get_logger().info(f"Proceeding to waypoint {self.wp_idx}")
                else:
                    self.phase = "RTL"  # Changed to RTL
                    self.get_logger().info("MISSION COMPLETE - RETURNING TO LAUNCH")

        elif self.phase == "FENCE_OUT":
            tx, ty = self.target
            vx = self._clamp(self.kp_x_base * (tx - x), self.vx_max * self.speed_factor)
            vy = self._clamp(self.kp_y_base * (ty - y), self.vy_max * self.speed_factor)
            self._sp_mixed(vx, vy, nan, self.z_ref)

            fence_error = math.sqrt((tx - x)**2 + (ty - y)**2)
            if fence_error < self.track_tol + 0.1:  # 10cm margin
                self.phase = "FENCE_BACK"
                self.get_logger().info("FENCE BOUNDARY REACHED")

        elif self.phase == "FENCE_BACK":
            bx, by = self.breach
            vx = self._clamp(self.kp_x_base * (bx - x), self.vx_max * self.speed_factor)
            vy = self._clamp(self.kp_y_base * (by - y), self.vy_max * self.speed_factor)
            self._sp_mixed(vx, vy, nan, self.z_ref)

            return_error = math.sqrt((bx - x)**2 + (by - y)**2)
            if return_error < self.track_tol + 0.1:  # 10cm margin
                self.phase = "MISSION"
                self.wp_idx = self.saved_wp
                self.get_logger().info("FENCE RECOVERY COMPLETE")

        elif self.phase == "AVOID":
            if (self.pos.xy_valid and self.pos.z_valid and 
                (x, y, z) not in self.obstacle_positions):
                self.obstacle_positions.append((x, y, z))
            
            if self.avoid_direction is None:
                self.avoid_direction = self.determine_escape_direction()
            
            vx, vy, vz = 0.0, 0.0, 0.0
            if self.avoid_direction == 'back':
                vy = -self.escape_velocity
            elif self.avoid_direction == 'front':
                vy = self.escape_velocity
            elif self.avoid_direction == 'left':
                vx = -self.escape_velocity
            elif self.avoid_direction == 'right':
                vx = self.escape_velocity
            elif self.avoid_direction == 'up':
                vz = -self.escape_velocity
            
            if self.avoid_direction == 'up':
                self._sp_mixed(0, 0, vz, nan)
            else:
                self._sp_mixed(vx, vy, nan, self.z_ref)
            
            if self.all_clear():
                self.phase = self.previous_phase
                self.avoid_direction = None
                self.get_logger().info("PATH CLEAR")

        elif self.phase == "RTL":
            if not self.rtl_cmd_sent:
                self.rtl()
                self.rtl_cmd_sent = True
                self.rtl_start_time = self.get_clock().now()
            
            # Check RTL progress
            if self.stat.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_RTL:
                if not self.rtl_state_logged:
                    self.get_logger().info("RTL in progress...")
                    self.rtl_state_logged = True
                
                # Check timeout (120 seconds)
                elapsed = (self.get_clock().now() - self.rtl_start_time).nanoseconds / 1e9
                if elapsed > 120.0:  # 2 minutes timeout
                    self.get_logger().error("RTL TIMEOUT - Forcing land")
                    self.land()
                    self.phase = "LAND"
            elif self.stat.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                self.get_logger().info("Landing...")
                self.phase = "LAND"
            
            # Check if landed
            if abs(z) < 0.5:  # 50cm from ground
                self.get_logger().info("RTL COMPLETE")
                self.phase = "RTL_COMPLETE"
                self.mission_active = False
                    
        elif self.phase == "SAFETY":
            if self.loop_counter % 20 == 0:
                self.get_logger().error(f"SAFETY: {self.safety_reason}")
                
            if self.stat.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_RTL:
                self.get_logger().info("RTL in progress...")
            elif self.stat.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                self.get_logger().info("Landing in progress...")
                self.phase = "LAND"

# ─── entry point ──────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = FenceMissionActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt - shutting down")
    finally:
        # Ensure video recorder is stopped
        if node.recording_active:
            node.video_recorder.stop_recording()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()  
