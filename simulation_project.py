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

DRAWER_INDEX = 1 # index 1 works perfectly

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

def find_gray_object(rgb_frame):
    if rgb_frame is None:
        return None
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([179, 30, 190]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    h_frame, w_frame = rgb_frame.shape[:2]
    frame_center = (w_frame / 2, h_frame / 2)

    closest_object = None
    min_distance = float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)

        center_x = int(x + w/2)
        center_y = int(y + h/2)

        distance = np.sqrt(
            (center_x - frame_center[0]) ** 2 +
            (center_y - frame_center[1]*0.6) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            closest_object = {
                "bbox": (x, y, w, h),
                "center": (center_x, center_y),
                "distance": distance
            }
    if closest_object is not None:
        x, y, w, h = closest_object["bbox"]
        cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb_frame, "Target", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return closest_object

def find_multiple_gray_objects(rgb_frame):
    if rgb_frame is None:
        return None
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([179, 30, 190]))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)

        all_is_detected = False

        if area > 900:
            table_c = (int(x + w/2), int(y + h/2))
            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(rgb_frame, f"Table", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            all_is_detected = True
            print("Table detected")
        else:
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            detected_objects.append({
                "id": i,
                "bbox": (x, y, w, h),
                "center": (center_x, center_y)
            })

            detected_objects.sort(key=lambda item: item["center"][1])

            for index, obj in enumerate(detected_objects):
                object_number = index + 1
                
                x, y, w, h = obj["bbox"]
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(rgb_frame, f"Object {object_number}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if all_is_detected:
        print(f"{len(detected_objects)} objects detected.")
        return table_c, detected_objects, all_is_detected
    else:
        return None, None, all_is_detected

def main():
    print("Press Ctrl+C to stop\n")

    Kp_pan      = 3
    Kp_tilt     = 0.04
    Kp_angle    = 5.0 / math.pi
    Kp_forward  = 2.0
    Kp_lift     = 5.0
    Kp_yaw      = 0.8
    Kp_pitch    = 0.8
    Kp_arm      = 10.0

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
            "gripper_open": 0.4,
            "arm_out": 0.0,
        })
        pre_grip_pose_object = StateController(controller, {
            "wrist_roll_counterclockwise": 0.0,
            "gripper_open": 0.4,
        })
        pre_grip_pose = StateController(controller, {
            "wrist_roll_counterclockwise": 1.6,
            "gripper_open": 0.4,
        })
        start_roll = False

        pull_back_drawer = StateController(controller, {
            "arm_out": 0.05,
        })
        start_complete_pull_back = False
        pull_finish = False

        high_pose = StateController(controller, {
            "lift_up": 0.9,
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 0.0,
            "wrist_pitch_up": 0.0,
            "arm_out": 0.0,
        })

        object_placed = False

        finish = False


        while True:
            # --- Loop setup ---
            t = controller.get_time()
            velocities = teleop.get_normalized_velocities() # manual override input
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
                    table, drawers, all_is_detected = find_multiple_gray_objects(rgb_frame)
                    if all_is_detected == False:
                        controller.set_velocities({"head_tilt_up": 0.1})
                    else:
                        print("All objects detected.")
                        drawer_centers = [obj["center"] for obj in drawers]
                        centroid_drawer = drawer_centers[DRAWER_INDEX] # Aim for a drawer
                    
                        depth = HEAD_CAMERA.get_depth(centroid_drawer)
                        cv2.circle(rgb_frame, centroid_drawer, 8, (0, 255, 0), -1)
                        if depth is not None:
                            cv2.putText(rgb_frame, f"{depth:.2f}m",
                            (centroid_drawer[0] + 10, centroid_drawer[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        cx_drawer, cy_drawer = centroid_drawer
                        cx, cy = table
                        frame_cx = rgb_frame.shape[1] / 2
                        frame_cy = rgb_frame.shape[0] / 2
                        error_x = (cx_drawer - frame_cx) / rgb_frame.shape[1]
                        error_y = (cy + frame_cy) / rgb_frame.shape[0]
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
                                if 0.8 <= horizontal_distance <= 1.0:
                                    in_zone = True
                            else:
                                if horizontal_distance < 0.75 or horizontal_distance > 0.95:
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
                                print(f"\rDist: {horizontal_distance:.2f}m  "
                                    f"Align: {alignment:.3f}  Auth: {travel_auth:.2f}    [moving]",
                                    end="", flush=True)
                                
                            cam_z = transforms.get_wrist_cam_T()[2, 3]
                            auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01) + 0.10)

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
                    target = find_gray_object(rgb_wrist)
                    centroid = target["center"] if target is not None else None

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
                        auto_velocities["lift_up"] = Kp_lift * (z*0.95 - (cam_z))

                        # Check transition conditions
                        angle_err = abs(-math.pi / 2 - angle_z)
                        lift_err = abs(z*0.95 - (cam_z))
                        print(f"\rLift Err: {lift_err:.3f}m")
                        print(f"\rz = {z:.3f}  cam_z = {cam_z:.3f}")
                        if stow_pose.is_at_goal() and \
                            angle_err < math.radians(5) and \
                            lift_err < 0.04:
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
                target = find_gray_object(rgb_frame)
                centroid = target["center"] if target is not None else None

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
                        distance_error = distance - 0.12
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

                if start_complete_pull_back:

                    #print(high_pose.get_current_state())
                    print(f"\rLift: {high_pose.get_current_state()['lift_up']:.3f}, Arm: {high_pose.get_current_state()['arm_out']:.3f}", end="", flush=True)
                    #velocities = merge_proportional(velocities, {"gripper_open": 0.5})
                    #velocities = merge_proportional(velocities, {"arm_out": -0.3})
                    #velocities = merge_proportional(velocities, {"wrist_roll_counterclockwise": 0.01})
                    auto_velocities["gripper_open"] = 0.5
                    auto_velocities["arm_out"] = -0.4
                    #auto_velocities["wrist_pitch_up"] = 0.05 ##################################################################
                    velocities = merge_proportional(velocities, auto_velocities)
                    if stow_pose.get_current_state()["arm_out"] < 0.03:
                        velocities = merge_proportional(velocities, stow_pose.get_command())
                    if stow_pose.is_at_goal() and pull_finish == False:
                        pull_finish = True
                        print("\nPull back complete.")
                    if pull_finish:
                        velocities = merge_proportional(velocities, high_pose.get_command())
                        if high_pose.is_at_goal():
                            phase = "object_align"
                            print(f"\nPhase: {phase}")
                else:
                    if pull_back_drawer.is_at_goal():
                        start_complete_pull_back = True
                        state = high_pose.get_current_state()
                        print("Reached pull back pose.")
                        print(f"\rLift: {state['lift_up']:.3f}  Arm: {state['arm_out']:.3f}    ", end="", flush=True)
                        
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
                        auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.07))

                        angle_err = abs(-math.pi / 2 - angle_z)
                        lift_err  = abs(z - (cam_z + 0.07))
                        print(f"\nLift Error: {lift_err}")
                        print(f"\nObject z: {z}, Cam z: {cam_z}")
                        if stow_pose.is_at_goal() and angle_err < math.radians(5) and lift_err < 0.01:
                            phase = "object_reach"
                            print(f"\nPhase: {phase}")

                    if rgb is not None:
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                if rgb is not None:
                    cv2.imshow("Project", rgb)

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
                        error_x = (cx - frame_cx) / rgb.shape[1]
                        error_y = (cy - frame_cy) / rgb.shape[0]
                        auto_velocities["wrist_yaw_counterclockwise"] = Kp_yaw * error_x
                        auto_velocities["wrist_pitch_up"] = -Kp_pitch * error_y
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                    distance = WRIST_CAMERA.get_depth((cx, cy))
                    if distance is not None:
                        distance_error = distance - 0.12
                        auto_velocities["arm_out"] = Kp_arm * distance_error
                        print(f"\rDist: {distance:.3f}m  Err: {distance_error:+.3f}m   ", end="", flush=True)
                        if abs(distance_error) < 0.04 and pre_grip_pose_object.is_at_goal():
                            phase = "object_grab"
                            print(f"\nPhase: {phase}")

                if rgb is not None:
                    cv2.imshow("Project", rgb)

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

                lift_threshold = state["lift_up"] + 0.2
                arm_threshold = state["arm_out"] + 0.24

                placing_angle = -0.45
                if DRAWER_INDEX == 0:
                    placing_angle = -0.3

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
                    
                    auto_velocities["wrist_pitch_up"] = 0.1
                    velocities = merge_proportional(velocities, auto_velocities)
                    velocities = merge_proportional(velocities, stow_pose.get_command())
                    print(stow_pose.get_current_state())

                else:

                    if placing_pose.is_at_goal():

                        print("Object placed")
                        object_placed = True
                        
                    else:
                        auto_velocities["arm_out"] = 0.03 ###########################################################
                        velocities = merge_proportional(velocities, placing_pose.get_command())
                        print(f"\nLift: {high_pose.get_current_state()['lift_up']:.3f}, Threshold: {lift_threshold:.3f}")
                        print(f"\nArm: {high_pose.get_current_state()['arm_out']:.3f}, Threshold: {arm_threshold:.3f}")


            # ----------------------------------------------------------------
            elif phase == "push_drawer":
                rgb = HEAD_RGB_CAMERA.get_frame()
                if rgb is not None:
                    cv2.imshow("Project", rgb)
                
                lift_threshold = state["lift_up"] + 0.1
                arm_threshold = arm_push_extension - 0.03


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
