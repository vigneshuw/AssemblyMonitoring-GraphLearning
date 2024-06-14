import os
import sys
import time
import pickle
import torch
import numpy as np
import pandas as pd
import utils_graph as gutils
from torch_geometric.data import HeteroData


class GraphConstructor:

    def __init__(self, data_params: dict = None, assembly: str = None) -> None:

        # Initialize the variables
        self.data_params = data_params
        self.assembly = assembly

        # Items for later initialization
        self.training_videos_fullpath = None
        self.training_annotations_fullpath = None
        self.processed_objects_path = None
        self.window_size = None
        self.overlap = None
        self.selected_class = None
        # Constructed Graphs
        self.data_list = []

    def load_data_and_labels(self, data_type="training"):

        if self.data_params is None:
            print("Cannot be used to load data, as 'data_params' is `None`")
            return None

        assembly_operation = self.data_params[self.assembly]

        # Training data location
        training_videos_dir = assembly_operation["data_dir"][data_type]
        training_video_names = [os.path.join(training_videos_dir, cycle) for cycle in os.listdir(training_videos_dir) if
                                cycle.split(".")[-1] == "mp4"]
        training_annotation_names = ["human_" + cycle.split("/")[-1][:-4] + ".csv" for cycle in training_video_names]

        # Assert that all the training data is labelled
        assert len(training_video_names) == len(training_annotation_names), "The labelling and the data for training " \
                                                                            "does not match"
        # Get the full path of the training data
        self.training_videos_fullpath = [os.path.join(training_videos_dir, cycle_name) for cycle_name in
                                         training_video_names]
        self.training_annotations_fullpath = [os.path.join(training_videos_dir, annotation_name) for annotation_name in
                                              training_annotation_names]
        sys.stdout.write("Videos and annotations loaded\n")

    def process_single_video(self, processed_objects_path, video_path, annotation_path, window_size, overlap,
                             selected_class):

        # Ensure the selected_class object is a list
        assert isinstance(selected_class, list), "The selected class object should be list"

        self.training_videos_fullpath = [video_path]
        self.training_annotations_fullpath = [annotation_path]

        # The path will be reconstructed later
        temp_len = len(processed_objects_path.split(os.sep)[-1])
        processed_objects_path = processed_objects_path[0:-temp_len]

        self.construct_graphs(processed_objects_path, window_size, overlap, selected_class)

        return self.data_list

    def construct_graphs(self, processed_objects_path: str, window_size: int, overlap: int, selected_class: list):

        # Initialize
        self.window_size = window_size
        self.overlap = overlap
        self.selected_class = selected_class

        # Make a final assertion to ensure they match
        start_time = time.time()
        self.data_list = []
        all_tasks = pd.DataFrame()
        for video_name, annotation_name in zip(self.training_videos_fullpath, self.training_annotations_fullpath):
            aug_video_name = video_name.split(os.path.sep)[-1].split(".")[0]
            aug_annotation_name = ("_".join(annotation_name.split(os.path.sep)[-1].split("_")[1:])).split(".")[0]
            assert aug_video_name == aug_annotation_name, "The annotations and the cycle do not match. Please check " \
                                                          "training directory"
        # Load into DataFrame
        y_list = []
        # Initialize frame_data_mapping dictionary
        for annotation_path in self.training_annotations_fullpath:
            y_list.append(pd.read_csv(annotation_path, header=0, names=["task", "startTime", "endTime"]))
        # Process the DataFrame
        video_counter = 0
        total_frame_window_count = 0  # Initialize the counter for total frame windows

        for video, annotation in zip(self.training_videos_fullpath, y_list):
            frame_data_mapping = {}
            y = []
            combined_y = None

            processed_y = gutils.modify_annotation_data(input_video_path=video, annotation_list=annotation)
            y.append(processed_y)
            # Stack the y together
            imp_cols = ["video", "task", "startTime", "endTime", "start_frame", "end_frame"]
            for index, df_y in enumerate(y):
                # Set the video ID
                df_y["video"] = video_counter

                if index == 0:
                    combined_y = df_y[imp_cols]
                    index += 1
                else:
                    combined_y = pd.concat([combined_y, df_y[imp_cols]], axis=0, ignore_index=True)
            all_tasks = pd.concat([all_tasks, combined_y])
            sys.stdout.write(f"Total number of task instances: {combined_y['task'].count()}\n")
            # print("\n")
            video_counter += 1
            combined_y["start_frame"] = combined_y["start_frame"].astype(int)
            combined_y["end_frame"] = combined_y["end_frame"].astype(int)
            print("video:", video)

            for index, task in combined_y.iterrows():
                seq = list(range(int(task['start_frame']), int(task['end_frame']) + 1))
                frame_window_list = gutils.get_shifting_window(seq, self.window_size, self.overlap)

                for frame_window in frame_window_list:
                    frame_metadata_list = []  # To store the frame metadata for each frame in the frame window
                    labels, tasks = gutils.load_labels(combined_y, frame_window)
                    adjusted_frame_window = []  # List to store adjusted frame window

                    for frames in frame_window:
                        frame_path = os.path.join(processed_objects_path, video.split('/')[-1][:-4], str(frames)
                                                  + '.pickle')
                        if os.path.exists(frame_path):
                            with open(frame_path, 'rb') as handle:
                                frame_metadata = pickle.load(handle)
                                # Convert to a tensor if it is a numpy array
                                if isinstance(frame_metadata["features"], np.ndarray):
                                    frame_metadata['features'] = torch.Tensor(frame_metadata['features'])
                                frame_metadata['features'] = frame_metadata['features'].cpu()
                                frame_metadata_list.append(
                                    frame_metadata)  # store corresponding metadata for each frame window
                                adjusted_frame_window.append(frames)  # Add the frame to the adjusted frame window

                    if len(adjusted_frame_window) == 0:
                        continue
                    frame_data_mapping[tuple(adjusted_frame_window)] = {"metadata": frame_metadata_list, "task": tasks}

                    # Associate frame_window with frame_metadata list and task labels

            # Number of frame_windows
            print("Number of frame_windows per video:", len(frame_data_mapping))
            torch.cuda.empty_cache()

            for index, (frame_window, frame_window_data) in enumerate(
                    frame_data_mapping.items()):  # inside all frame windows in a specific video
                #         print(frame_window)
                total_object_count = 0
                # total_object_count_classes =0
                original_frame_window_features = []  # Initialize the list to store original features for each frame window in the current video
                frame_window_features = []  # Initialize the list to store modified features for each frame window in the current video
                frame_window_classes = []  # Initialize the list to store classes for each frame window
                frame_window_boxes = []  # Initialize the list to store boxes for each frame window
                frame_window_tasks = []
                frame_window_tasks.append(frame_window_data['task'])
                total_frame_window_count += 1
                frame_window_metadata_list = frame_window_data[
                    "metadata"]  # all frame metadata for all the frames within a specific frame window
                # inside all frames within a specific frame window within a specific video:
                for frame_window_idx, frame_window_metadata in enumerate(frame_window_metadata_list):
                    frame_features = frame_window_metadata[
                        "features"]  # features of all the frames within a specific frame window
                    frame_boxes = frame_window_metadata[
                        "boxes"]  # bounding box coordinates of all the frames within a specific frame window
                    frame_classes = frame_window_metadata[
                        "class"]  # classes of all the frames within a specific frame window

                    total_object_count += len(frame_boxes)
                    frame_window_classes.append(frame_classes)
                    #         print(frame_window_classes)
                    frame_window_boxes.append(frame_boxes)
                    original_frame_window_features.append(frame_features)

                    for object_idx, object_feature in enumerate(
                            frame_features):  # inside all objects within a specific frame
                        max_pool = torch.nn.MaxPool2d(kernel_size=(7, 7))
                        pooled_features_per_object = max_pool(object_feature)
                        averaged_features_per_object = torch.mean(pooled_features_per_object, dim=(-2, -1))
                        frame_window_features.append(averaged_features_per_object)

                frame_window_features = torch.stack(frame_window_features, dim=0)
                row_sums = frame_window_features.sum(dim=1, keepdim=True)
                frame_window_features = frame_window_features / row_sums
                labels = gutils.encode_onehot(all_tasks, frame_window_tasks)
                labels = torch.LongTensor(np.where(labels)[1])

                adj_spatial, adj_temporal = gutils.construct_iou_adjacency_matrix(
                    frame_window_classes, frame_window_boxes, total_object_count, self.selected_class)
                edge_index_spatial = adj_spatial.nonzero().t()
                edge_index_spatial = edge_index_spatial.to(torch.long)
                edge_weight_spatial = adj_spatial[edge_index_spatial[0], edge_index_spatial[1]]
                edge_weight_spatial = edge_weight_spatial.to(torch.float)

                edge_index_temporal = adj_temporal.nonzero().t()
                edge_index_temporal = edge_index_temporal.to(torch.long)
                edge_weight_temporal = adj_temporal[edge_index_temporal[0], edge_index_temporal[1]]
                edge_weight_temporal = edge_weight_temporal.to(torch.float)

                data = HeteroData(frame_window={'x': frame_window_features})
                data['frame_window'].y = labels
                data['frame_window', 'spatial', 'frame_window'].edge_index = edge_index_spatial
                data['frame_window', 'spatial', 'frame_window'].edge_weight = edge_weight_spatial
                data['frame_window', 'temporal', 'frame_window'].edge_index = edge_index_temporal
                data['frame_window', 'temporal', 'frame_window'].edge_weight = edge_weight_temporal

                self.data_list.append(data)

        num_data_objects = len(self.data_list)
        print(f"Number of Data objects in the list: {num_data_objects}")

        print("--- %s seconds ---" % (time.time() - start_time))