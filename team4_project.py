"""
Stretch Toolkit boilerplate — starting point for new scripts.
"""
import math
from stretch_toolkit import ( controller, teleop, merge_proportional, locate_object, BACKEND_NAME, HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA, HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, StateController, RobotTransforms)
import stretch_toolkit.input as inp
import time
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

def find_red_object(rgb_frame):
    if rgb_frame is None:
        return None
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    #mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    #mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return None

def find_green_object(rgb_frame):
    if rgb_frame is None:
        return None
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 60, 140]), np.array([85, 220, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return None

def main():
    print("Press Ctrl+C to stop\n")

    Kp_pan      = 1.0
    Kp_tilt     = 1.0
    Kp_angle    = 5.0 / math.pi
    Kp_forward  = 2.0
    Kp_lift     = 5.0
    Kp_yaw      = 0.8
    Kp_yaw_object = 0.8
    Kp_pitch_object = 0.5
    Kp_pitch    = 0.8
    Kp_arm      = 10.0

    GRIPPER_CAM_DY_PX  = 75    # px the object should sit BELOW frame center (reach)
    LIFT_HEIGHT_OFFSET = 0.035  # m of extra lift height at the align settle point

    in_zone = False
    phase = "approach"
    print(f"Phase: {phase}")

    try:

        transforms = RobotTransforms(controller)
        pre_camera_pose = StateController(controller, {
            "head_tilt_up": -1.2,              # radians
        })
        reached_initial_camera_pose = False

        stow_pose = StateController(controller, {
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 0.0,
            "wrist_pitch_up": 0.0,
            "gripper_open": 4.0,
            "arm_out": 0.0,
        })
        pre_grip_pose_object = StateController(controller, {
            "wrist_roll_counterclockwise": 0.0,
            "gripper_open": 4.0,
        })
        pre_grip_pose = StateController(controller, {
            "wrist_roll_counterclockwise": 1.6,
            "gripper_open": 4.0,
        })
        start_roll = False

        pull_back_drawer = StateController(controller, {
            "arm_out": 0.13,
        })
        start_complete_pull_back = False
        pull_finish = False
        drawer_released = False

        high_pose = StateController(controller, {
            "lift_up": 0.9,
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 0.0,
            "wrist_pitch_up": 0.0,
            "arm_out": 0.0,
        })

        object_placed = False

        finish = False

        grab_start_time = None
        grab_debug_print_time = 0.0

        while True:
            # --- Loop setup ---
            t = controller.get_time()
            velocities = {}
            #velocities = teleop.get_normalized_velocities() # manual override input
            auto_velocities = {} # autonomous commands

            # ----------------------------------------------------------------
            if phase == "approach":
                #initial = pre_camera_pose.get_current_state()
                while not reached_initial_camera_pose:
                    controller.set_velocities(pre_camera_pose.get_command())
                    #progress = pre_camera_pose.get_progress(initial)
                    #print ( f"Progress: {progress['head_tilt_up']:.2f}")
                    if pre_camera_pose.is_at_goal():
                         print("Reached initial camera pose.")
                         reached_initial_camera_pose = True
                
                rgb = HEAD_RGB_CAMERA.get_frame()

                if rgb is not None:

                    h_frame, w_frame = rgb.shape[:2]
                    rgb_frame = rgb[:h_frame*2//3, :]
                    centroid_drawer = find_green_object(rgb_frame)

                    if centroid_drawer is not None:
                    
                        depth = HEAD_CAMERA.get_depth(centroid_drawer)
                        cv2.circle(rgb_frame, centroid_drawer, 8, (0, 255, 0), -1)
                        if depth is not None:
                            cv2.putText(rgb_frame, f"{depth:.2f}m",
                            (centroid_drawer[0] + 10, centroid_drawer[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        cx_drawer, cy_drawer = centroid_drawer
                        frame_cx = rgb_frame.shape[1] / 2
                        frame_cy = rgb_frame.shape[0] / 2
                        error_x = (cx_drawer - frame_cx) / rgb_frame.shape[1]
                        error_y = (cy_drawer - frame_cy) / rgb_frame.shape[0]
                        auto_velocities["head_pan_counterclockwise"]    = -Kp_pan * error_x
                        auto_velocities["head_tilt_up"]                 = -Kp_tilt * error_y

                        _, obj2base_T = locate_object((cx_drawer, cy_drawer), HEAD_CAMERA, transforms)
                        if obj2base_T is not None:
                            x, y, z = obj2base_T[0:3, 3]
                            angle_z = math.atan2(y, x)
                            horizontal_distance = math.sqrt(x**2 + y**2)

                            # Start raising lift toward object height early (10cm above align target)
                            cam_z = transforms.get_wrist_cam_T()[2, 3]
                            auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01) + 0.10)

                            # Update zone membership with hysteresis
                            if not in_zone:
                                if 0.6 <= horizontal_distance <= 0.8:
                                    in_zone = True
                            else:
                                if horizontal_distance < 0.55 or horizontal_distance > 0.75:
                                    in_zone = False
                            
                            # Choose behavior based on zone
                            if in_zone:
                                # Rotate to flank angle (-pi/2 = object to the right)
                                angle_error = -math.pi / 2 - angle_z
                                auto_velocities["base_counterclockwise"] = Kp_angle * angle_error
                                auto_velocities["base_forward"] = 0.0
                                print(f"\rDist: {horizontal_distance:.2f}m  "
                                    f"Angle Err: {math.degrees(angle_error) :+6.1f}deg    [FLANK]",
                                    end="", flush=True)
                                # Transition to align once roughly flanked
                                if abs(angle_error) < math.radians(5):
                                    #cv2.destroyWindow("Head RGB")
                                    phase = "align"
                                    print(f"\nPhase: {phase}")
                            else:
                                # Face and drive toward object
                                auto_velocities["base_counterclockwise"] = -Kp_angle * angle_z
                                alignment = 1.0 - abs(angle_z) / math.pi
                                travel_auth = max(0.0, min(1.0, (alignment - 0.9) / 0.1))
                                auto_velocities["base_forward"] = Kp_forward * horizontal_distance * travel_auth
                                print(f"\rDist: {horizontal_distance:.2f}m  Align: {alignment:.3f}  Auth: {travel_auth:.2f}    [moving]", end="", flush=True)

                if rgb is not None:
                    cv2.imshow("Project", rgb_frame)

                # --- Send commands ---
                velocities = merge_proportional(velocities, stow_pose.get_command())
                velocities = merge_proportional(velocities, auto_velocities)

            # ----------------------------------------------------------------
            elif phase == "align":

                rgb_wrist = WRIST_RGB_CAMERA.get_frame()
                if rgb_wrist is not None:
                    h_frame, w_frame = rgb_wrist.shape[:2]
                    centroid = find_green_object(rgb_wrist)
                    if centroid is not None:

                        cx, cy = centroid
                        _, obj2base_T = locate_object((cx, cy), WRIST_CAMERA, transforms)
                        if obj2base_T is not None:
                            x, y, z = obj2base_T[0:3, 3]
                            angle_z = math.atan2(y, x)
                            cam_z = transforms.get_wrist_cam_T()[2, 3]

                            # Rotate base toward final flank target (3 deg bias for gripper offset)
                            auto_velocities["base_counterclockwise"] = \
                                Kp_angle * (-math.pi / 2 - angle_z + math.radians(3))
                            # Move lift until object is level with wrist camera
                            auto_velocities["lift_up"] = Kp_lift * (z - (cam_z - 0.1))

                            # Check transition conditions
                            angle_err = abs(-math.pi / 2 - angle_z)
                            lift_err = abs(z - (cam_z - 0.1))
                            print(f"\rLift Err: {lift_err:.3f}m")
                            print(f"\rz = {z:.3f}  cam_z = {cam_z:.3f}")
                            if stow_pose.is_at_goal() and \
                                angle_err < math.radians(5) and \
                                lift_err < 0.03:
                                start_roll = True
                            if pre_grip_pose.is_at_goal():
                                phase = "reach"
                                print(f"\nPhase: {phase}")
                        
                        if rgb_wrist is not None:                        
                            cv2.circle(rgb_wrist, (cx, cy), 10, (0, 255, 0), -1)

                if rgb_wrist is not None:
                    cv2.imshow("Project", rgb_wrist)
                
                if start_roll:
                    velocities = merge_proportional(velocities, pre_grip_pose.get_command())
                else:
                    velocities = merge_proportional(velocities, auto_velocities)
                    velocities = merge_proportional(velocities, stow_pose.get_command())
            
            # ----------------------------------------------------------------
            elif phase == "reach":
                """
                while not reached_pre_grip_pose:
                    controller.set_velocities(pre_grip_pose.get_command())
                    if pre_grip_pose.is_at_goal():
                         print("Reached pre grip pose.")
                         reached_pre_grip_pose = True"""

                rgb = WRIST_RGB_CAMERA.get_frame()
                h_frame, w_frame = rgb.shape[:2]
                rgb_frame = rgb[:h_frame*2//3, w_frame//3:w_frame*2//3]
                centroid = find_green_object(rgb_frame)

                if centroid is not None and rgb_frame is not None:
                    centroid_x, centroid_y = centroid

                    # Servo wrist to keep object centered
                    if rgb_frame is not None:
                        frame_cx = rgb_frame.shape[1] / 2
                        frame_cy = rgb_frame.shape[0] / 2
                        error_x = (centroid_x - frame_cx) / rgb_frame.shape[1]
                        error_y = (centroid_y*0.6 - frame_cy) / rgb_frame.shape[0]
                        auto_velocities["wrist_yaw_counterclockwise"] = Kp_yaw * error_y
                        auto_velocities["wrist_pitch_up"] = -Kp_pitch * error_x
                        cv2.circle(rgb_frame, (centroid_x, centroid_y), 10, (0, 255, 0), -1)
                        print(f"\rError X: {error_x:.3f}  Error Y: {error_y:.3f}    ", end="", flush=True)

                    # Extend arm based on depth
                    distance = WRIST_CAMERA.get_depth((centroid_x, centroid_y))
                    if distance is not None:
                        distance_error = distance - 0.16
                        auto_velocities["arm_out"] = Kp_arm * distance_error
                        print(f"\rDepth: {distance:.3f}m  Error: {distance_error:+.3f}m     ", end="", flush=True)
                        if abs(distance_error) < 0.08:
                            arm_push_extension = stow_pose.get_current_state()['arm_out']
                            phase = "grab"
                            print(f"\nPhase: {phase}")
                
                if rgb_frame is not None:
                    cv2.imshow("Project", rgb_frame)
                
                velocities = merge_proportional(velocities, auto_velocities)

            # ----------------------------------------------------------------
            elif phase == "grab":
                
                rgb = HEAD_RGB_CAMERA.get_frame()
                if rgb is not None:
                    cv2.imshow("Project", rgb)

                if grab_start_time is None:
                    grab_start_time = time.time()
                    print("\n[grab] entered — waiting for gripper to close", flush=True)

                gripper_delay = 1.5  # seconds
                elapsed = time.time() - grab_start_time

                if start_complete_pull_back:

                    arm_now  = stow_pose.get_current_state()["arm_out"]
                    stow_at_goal = stow_pose.is_at_goal()

                    # Per-joint debug every ~1 s
                    now = time.time()
                    if now - grab_debug_print_time >= 1.0:
                        grab_debug_print_time = now
                        stow_state = stow_pose.get_current_state()
                        stow_targets = {
                            "wrist_roll_counterclockwise": 0.0,
                            "wrist_yaw_counterclockwise":  0.0,
                            "wrist_pitch_up":              0.0,
                            "gripper_open":                4.0,
                            "arm_out":                     0.0,
                        }
                        print(f"\n[grab] stow_at_goal={stow_at_goal}  pull_finish={pull_finish}  arm={arm_now:.3f}", flush=True)
                        for joint, target in stow_targets.items():
                            cur = stow_state.get(joint, float('nan'))
                            print(f"  {joint:35s} cur={cur:.3f}  target={target:.3f}  err={abs(cur-target):.3f}", flush=True)

                    auto_velocities["gripper_open"] = 0.5
                    auto_velocities["arm_out"] = -0.4

                    if not drawer_released and stow_pose.get_current_state()["gripper_open"] > 1.2:
                        drawer_released = True
                        print("\n[grab] drawer handle released, now moving toward stow_pose", flush=True)

                    if drawer_released:
                        velocities = merge_proportional(velocities, stow_pose.get_command())
                    velocities = merge_proportional(velocities, auto_velocities)

                    if drawer_released and stow_at_goal and not pull_finish:
                        pull_finish = True
                        print("\n[grab] stow_pose reached — pull_finish = True", flush=True)
                    if pull_finish:
                        velocities = merge_proportional(velocities, high_pose.get_command())
                        high_state = high_pose.get_current_state()
                        print(f"\r[grab] driving high_pose — lift={high_state['lift_up']:.3f}  "
                              f"arm={high_state['arm_out']:.3f}  at_goal={high_pose.is_at_goal()}   ",
                              end="", flush=True)
                        if high_pose.is_at_goal():
                            phase = "object_align"
                            print(f"\nPhase: {phase}")

                elif elapsed < gripper_delay:
                    # Phase 1: close gripper only, don't move arm yet
                    auto_velocities["gripper_open"] = -1.5
                    velocities = merge_proportional(velocities, auto_velocities)
                    print(f"\r[grab] closing gripper {elapsed:.1f}/{gripper_delay:.1f}s  "
                          f"gripper={stow_pose.get_current_state()['gripper_open']:.3f}   ",
                          end="", flush=True)

                else:
                    # Phase 2: gripper closed, now pull back
                    pb_at_goal = pull_back_drawer.is_at_goal()
                    pb_arm = pull_back_drawer.get_current_state().get('arm_out', float('nan'))
                    print(f"\r[grab] pulling back — arm={pb_arm:.3f}  pull_back_at_goal={pb_at_goal}   ",
                          end="", flush=True)
                    if pb_at_goal:
                        start_complete_pull_back = True
                        state = high_pose.get_current_state()
                        print(f"\n[grab] pull_back_drawer reached — start_complete_pull_back = True  "
                              f"lift={state['lift_up']:.3f}  arm={state['arm_out']:.3f}", flush=True)
                    else:
                        auto_velocities["gripper_open"] = -1.5
                        auto_velocities["arm_out"] = -0.4
                        velocities = merge_proportional(velocities, auto_velocities)

            # ----------------------------------------------------------------
            elif phase == "object_align":
                rgb = WRIST_RGB_CAMERA.get_frame()
                centroid = find_red_object(rgb)

                if centroid is not None:
                    cx, cy = centroid

                    _, obj2base_T = locate_object((cx, cy), WRIST_CAMERA, transforms)
                    if obj2base_T is not None:
                        x, y, z = obj2base_T[0:3, 3]
                        angle_z = math.atan2(y, x)
                        cam_z = transforms.get_wrist_cam_T()[2, 3]

                        auto_velocities["base_counterclockwise"] = Kp_angle * (-math.pi / 2 - angle_z + math.radians(3))
                        auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01) + LIFT_HEIGHT_OFFSET)
                        angle_err = abs(-math.pi / 2 - angle_z)
                        lift_err  = abs(z - (cam_z + 0.01) + LIFT_HEIGHT_OFFSET)

                        if stow_pose.is_at_goal() and angle_err < math.radians(5) and lift_err < 0.03:
                            phase = "object_reach"
                            print(f"\nPhase: {phase}")

                    if rgb is not None:
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                if rgb is not None:
                    cv2.imshow("Wrist RGB", rgb)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, stow_pose.get_command())

            # --------------------state--------------------------------------------
            elif phase == "object_reach":
                rgb = WRIST_RGB_CAMERA.get_frame()
                centroid = find_red_object(rgb)

                if centroid is not None:
                    cx, cy = centroid

                    if rgb is not None:
                        frame_cx = rgb.shape[1] / 2
                        frame_cy = rgb.shape[0] / 2
                        # Aim the object slightly below frame center so the
                        # fingertips (not the lens) line up with the target.
                        target_cy = frame_cy + GRIPPER_CAM_DY_PX
                        error_x = (cx - frame_cx) / rgb.shape[1]
                        error_y = (cy - target_cy) / rgb.shape[0]
                        auto_velocities["wrist_yaw_counterclockwise"] = -Kp_yaw * error_x
                        auto_velocities["wrist_pitch_up"] = -Kp_pitch * error_y
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                    distance = WRIST_CAMERA.get_depth((cx, cy))
                    if distance is not None:
                        distance_error = distance - 0.165
                        auto_velocities["arm_out"] = Kp_arm * distance_error
                        print(f"\rDist: {distance:.3f}m  Err: {distance_error:+.3f}m   ", end="", flush=True)
                        if abs(distance_error) < 0.02 and pre_grip_pose_object.is_at_goal():
                            cv2.destroyAllWindows()
                            phase = "object_grab"
                            print(f"\nPhase: {phase}")

                if rgb is not None:
                    cv2.imshow("Wrist RGB", rgb)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, pre_grip_pose_object.get_command())

            # ----------------------------------------------------------------
            elif phase == "object_grab":
                
                rgb = HEAD_RGB_CAMERA.get_frame()
                if rgb is not None:
                    cv2.imshow("Project", rgb)

                auto_velocities["gripper_open"] = -1.5
                if stow_pose.get_current_state()["gripper_open"] < 0.1:
                    velocities = merge_proportional(velocities, high_pose.get_command())
                    if high_pose.get_current_state()["lift_up"] > 0.8 and high_pose.get_current_state()["arm_out"] < 0.05:
                        phase = "place_object"
                        print(f"\nPhase: {phase}")
                velocities = merge_proportional(velocities, auto_velocities)

            # ----------------------------------------------------------------
            elif phase == "place_object":

                rgb = HEAD_RGB_CAMERA.get_frame()
                if rgb is not None:
                    cv2.imshow("Project", rgb)

                lift_threshold = state["lift_up"] + 0.13
                arm_threshold = state["arm_out"] + 0.2

                placing_angle = -0.45

                placing_pose = StateController(controller, {
                    "wrist_pitch_up": placing_angle,
                    "lift_up": lift_threshold,
                    "arm_out": arm_threshold,
                })

                if object_placed:
                    if stow_pose.is_at_goal():
                        phase = "push_drawer"
                        print(f"\nPhase: {phase}")
                        print(f"\nArm Push Extension: {arm_push_extension:.3f}")

                    #auto_velocities["wrist_pitch_up"] = 0.1
                    #velocities = merge_proportional(velocities, auto_velocities)
                    velocities = merge_proportional(velocities, stow_pose.get_command())
                    stow_state = stow_pose.get_current_state()
                    stow_targets = {
                        "wrist_roll_counterclockwise": 0.0,
                        "wrist_yaw_counterclockwise":  0.0,
                        "wrist_pitch_up":              0.0,
                        "gripper_open":                4.0,
                        "arm_out":                     0.0,
                    }
                    for joint, target in stow_targets.items():
                        cur = stow_state.get(joint, float('nan'))
                        print(f"  {joint:35s} cur={cur:.3f}  target={target:.3f}  err={abs(cur-target):.3f}", flush=True)

                else:

                    if placing_pose.is_at_goal():
                        print("Arrived at placing pose — opening gripper and moving to stow")
                        auto_velocities["gripper_open"] = 1.5
                        if stow_pose.get_current_state()['gripper_open'] > 0.4:
                            object_placed = True
                        velocities = merge_proportional(velocities, auto_velocities)
                    else:
                        #auto_velocities["arm_out"] = 0.03 ###########################################################
                        velocities = merge_proportional(velocities, placing_pose.get_command())
                        print(f"\nLift: {high_pose.get_current_state()['lift_up']:.3f}, Threshold: {lift_threshold:.3f}")
                        print(f"\nArm: {high_pose.get_current_state()['arm_out']:.3f}, Threshold: {arm_threshold:.3f}")

            # ----------------------------------------------------------------
            elif phase == "push_drawer":
                rgb = HEAD_RGB_CAMERA.get_frame()
                if rgb is not None:
                    cv2.imshow("Project", rgb)
                
                lift_threshold = state["lift_up"] + 0.04
                arm_threshold = arm_push_extension


                if high_pose.get_current_state()["lift_up"] < lift_threshold:

                    if high_pose.get_current_state()["arm_out"] > arm_threshold:
                        finish = True
                    else:
                        auto_velocities["arm_out"] = 0.5
                    
                else:
                    auto_velocities["lift_up"] = -0.7    
                
                if finish:
                    velocities = merge_proportional(velocities, stow_pose.get_command())
                    if stow_pose.is_at_goal():
                            phase = "done"
                            print(f"\nPhase: {phase}")
                else:
                    velocities = merge_proportional(velocities, auto_velocities)

            # ----------------------------------------------------------------
            elif phase == "done":
                pass # Hold position

            velocities = teleop.get_manual_override(velocities)
            controller.set_velocities(velocities)
            cv2.waitKey(1)
            time.sleep(1 / 30)  # 30 Hz

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
