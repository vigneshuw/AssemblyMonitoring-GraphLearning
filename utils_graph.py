import os
import pandas as pd
import cv2
import pickle
import torch
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def modify_annotation_data(input_video_path, annotation_list):
    """
    -TO DO: Convert the time information of actions from seconds to frames

    -Inputs/ Arguments:
    annotation_list: df with startTime & endTime into seconds
    fps: frame per seconds at which the input video was recorded

    -Outputs/ Returns:
    annotation_list: dataframe containing both second and frame information
    """
    new_df = pd.DataFrame(columns=["task", "start_frame", "end_frame"])
    cap = cv2.VideoCapture(input_video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    for i in range(len(annotation_list)):
        task_i = annotation_list['task'].iloc[i]
        start_i = pd.Series(pd.to_timedelta([annotation_list["startTime"].iloc[i]])).dt.total_seconds()
        start_i = start_i[0] * fps_video
        end_i = pd.Series(pd.to_timedelta([annotation_list["endTime"].iloc[i]])).dt.total_seconds()
        end_i = end_i[0] * fps_video
        new_df.loc[i] = [task_i, start_i, end_i]
    annotation_list["start_frame"] = new_df["start_frame"].to_numpy()
    annotation_list["end_frame"] = new_df["end_frame"].to_numpy()
    return annotation_list


def get_shifting_window(data_items: list, size: int, overlap: int) -> list:
    windows = []
    i = 0
    while i < len(data_items):
        if i + size <= len(data_items):
            windows.append(data_items[i:i + size])
        else:
            windows.append(data_items[i:])
            break
        i += size - overlap
    return windows


def encode_onehot(all_tasks: pd.DataFrame, frame_window_tasks):
    lb = LabelBinarizer()
    lb.fit(all_tasks["task"])
    classesID = list(lb.classes_)
    num_classes = len(classesID)

    # Create a text file with the ordered classes so that we can decode it afterwards
    with open(
            "classes.txt",'w'
    ) as output:
        for row in classesID:
            output.write(str(row) + "\n")
    y = lb.transform(frame_window_tasks)
    return y


def pickle_data(data, folder, filename):
    with open(os.path.join(folder, filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_data(folder, filename):
    with open(os.path.join(folder, filename), 'rb') as handle:
        data = pickle.load(handle)
        return data


def diag_block_mat(tList: list):
    blkXsize = [tbl.shape[1] for tbl in tList]
    outBlocks = []
    for i, tbl in enumerate(tList):
        tBefore = np.zeros((tbl.shape[0], sum(blkXsize[:i])))
        tAfter = np.zeros((tbl.shape[0], sum(blkXsize[i+1:])))
        outBlocks.append(np.hstack([tBefore, tbl, tAfter]))
    return np.vstack(outBlocks)


def load_labels(combined_y, frame_window):
    last_frame = frame_window[-1]
    ind = 0
    for start, end in zip(combined_y["start_frame"], combined_y["end_frame"]):
        if start <= last_frame <= end:
            break
        ind += 1

    task = combined_y["task"][ind]
    # Create a list of dictionaries
    data = [{"FrameWindow": frame_window, "Task": task}]
    # Create a dataframe
    df = pd.DataFrame(data)

    return df, task


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def construct_iou_adjacency_matrix(frame_window_classes, frame_window_boxes, total_object_count):
    # Initialize the adjacency matrix for the entire frame window
    iou_matrix_list = []
    class_indices = []
    total_classes = 0
    frame_indices = []

    # Iterate over each frame in the frame window #FW
    for frame_idx, (frame_classes, frame_boxes) in enumerate(zip(frame_window_classes, frame_window_boxes)):
        # Initialize the IOU matrix for the current frame
        iou_matrix = np.zeros((len(frame_boxes), len(frame_boxes)))
        for object_idx, frame_class in enumerate(frame_classes):
            continuous_object_idx = total_classes + object_idx
            class_indices.append((continuous_object_idx, frame_idx, frame_class, frame_boxes[object_idx]))
        total_classes += len(frame_classes)

        # Calculate IOU and update the IOU matrix for each pair of objects in the same frame #Frame
        for i, box1 in enumerate(frame_boxes):
            for j, box2 in enumerate(frame_boxes):
                if i == j:  # skip self loop connections
                    continue
                iou = bb_intersection_over_union(box1, box2)

                fc1 = frame_classes[i]
                fc2 = frame_classes[j]

                # Update the IOU matrix
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        iou_matrix_list.append(iou_matrix)

    # Accumulate the IOU matrix of the current frame into the adjacency matrix

    iou_intra_matrix = diag_block_mat(iou_matrix_list)

    iou_inter_matrix = np.zeros((total_object_count, total_object_count))

    for i, (object_idx_i, frame_idx_i, frame_class_i, frame_box_i) in enumerate(class_indices):
        frame_class_counts = {}
        for j, (object_idx_j, frame_idx_j, frame_class_j, frame_box_j) in enumerate(class_indices):

            if i == j:  # skip self loop connections
                continue
            if frame_idx_i == frame_idx_j:  # skip same frame connections
                continue
            if frame_class_i != frame_class_j:
                continue
            elif frame_class_i == 12 and frame_class_j != 12:
                continue
            elif frame_class_i != 12 and frame_class_j == 12:
                continue
            elif frame_class_i != 12 and frame_class_j != 12:
                continue
            else:
                iou = bb_intersection_over_union(frame_box_i, frame_box_j)
                iou_inter_matrix[i, j] = iou
                iou_inter_matrix[j, i] = iou

            if frame_class_j not in frame_class_counts:
                frame_class_counts[frame_class_j] = {}
            if frame_idx_j not in frame_class_counts[frame_class_j]:
                frame_class_counts[frame_class_j][frame_idx_j] = {'count': 1, 'object_ids': [object_idx_j],
                                                                  'bounding_boxes': [frame_box_j]}
            else:
                frame_class_counts[frame_class_j][frame_idx_j]['count'] += 1
                frame_class_counts[frame_class_j][frame_idx_j]['object_ids'].append(object_idx_j)
                frame_class_counts[frame_class_j][frame_idx_j]['bounding_boxes'].append(frame_box_j)

    iou_inter_matrix = torch.tensor(iou_inter_matrix, dtype=torch.float32)
    iou_intra_matrix = torch.tensor(iou_intra_matrix, dtype=torch.float32)

    return iou_intra_matrix, iou_inter_matrix

