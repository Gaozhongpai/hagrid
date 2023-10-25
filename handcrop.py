import cv2
import mediapipe as mp
import os
import math
import fnmatch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
from glob import glob

def rotate_image(image, angle, image_center, scale, xy_coords, dsize=224):
    angle = math.degrees(angle)
    # image_size = int(224 / scale)
    ## image_center [col, row]
    inter_flag = cv2.INTER_LINEAR  # cv2.INTER_CUBIC
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    xy_news = np.transpose(np.dot(rot_mat, np.transpose(xy_coords)))
    xy_news = (xy_news - np.array(image_center)) + dsize//2  # important
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=inter_flag, borderMode=cv2.BORDER_CONSTANT)
    img_pad = cv2.copyMakeBorder(
        result, dsize//2, dsize//2, dsize//2, dsize//2, cv2.BORDER_CONSTANT)
    image_center = (int(image_center[0]), int(image_center[1]))
    image_crop = img_pad[image_center[1]:image_center[1] +
                         dsize, image_center[0]:image_center[0]+dsize]
    return image_crop, xy_news

color_hand_joints = [[1.0, 0.0, 0.0],
					 [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
					 [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
					 [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
					 [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
					 [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = np.copy(image)

    marker_sz = 6
    line_wd = 3
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype(
            'int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype(
                'int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1,
                              0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=3)

root_dir = "../HanCo/handgesture/"
for root, dirs, files in os.walk(root_dir):
    
    colors = fnmatch.filter(files, "*.jpg")[:400]
    # depths = fnmatch.filter(files, "*_Depth_*.png")
    for filename in tqdm(colors):
        fpath = os.path.join(root, filename)

        # Load image using OpenCV
        image = cv2.imread(fpath)
        image_height, image_width, _ = image.shape

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        result = hands.process(image_rgb)

        # Check if hand landmarks are detected
        if result.multi_hand_landmarks:
            for k in range(len(result.multi_hand_landmarks)):

                scale = 224 / max(result.hand_rects_from_landmarks[k].height*image_height,
                                result.hand_rects_from_landmarks[k].width*image_width) * 0.85
                rot = 0  # result.hand_rects_from_landmarks[0].rotation
                center = (int(result.hand_rects_from_landmarks[k].x_center*image_width),
                        int(result.hand_rects_from_landmarks[k].y_center*image_height))

                # Get the hand landmarks
                hand_landmarks = result.multi_hand_landmarks[k]
                kpt = np.ones((21, 3), dtype=np.float32)
                for i, lmk in enumerate(hand_landmarks.landmark):
                    kpt[i, 0:2] = np.array([lmk.x*image_width, lmk.y*image_height])

                image_roate, kp_new = rotate_image(image, rot, center, scale, kpt)
                # sk_2d_img = draw_2d_skeleton(image_roate, kp_new)
                # # Draw hand landmarks on the image for visualization (optional)
                # mp_drawing.draw_landmarks(
                #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fpath_save = Path(fpath.replace("handgesture", "handgesture_crop").replace(".jpg", f"{k:02}.jpeg"))
                Path(fpath_save.parent).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(fpath_save), image_roate)
                #### for depth ######
                # stem = Path(fpath).stem.split(".")[0].replace("Color", "Depth")
                # depth_path = next((fdepth for fdepth in depths if stem in fdepth), None)
                # if depth_path:
                #     depth = cv2.imread(os.path.join(root, depth_path))
                #     depth_roate, _ = rotate_image(depth, rot, center, scale, kpt)
                #     cv2.imwrite(str(fpath_save).replace("Color", "Depth"), depth_roate)
        else:
            print(f"{fpath}: no hand")

# Close the MediaPipe Hands instance
hands.close()
