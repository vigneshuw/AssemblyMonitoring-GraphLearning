#%% Imports
import os
import numpy as np
import concurrent.futures
import sys
import yaml
import pickle
from graph_construction import GraphConstructor

#%% Configuration
# Load configuration
with open("graph_generator.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
video_paths = cfg['video_paths']
video_object_paths = cfg['video_object_paths']
# Graph Information
graph_window = cfg['graph']["window"]
graph_overlap = cfg['graph']["overlap"]
selected_class = cfg['graph']["selected_class"]
# List all the videos inside the object paths
video_object_locations = {}
for video_object_path in video_object_paths:
    video_object_locations[video_object_path] = os.listdir(video_object_path)

# Output path for saving the processed videos
output_path = os.path.join(cfg['output_path'], "GraphConstruction", cfg['graph']["type"], f"sc-{selected_class}",
                           f"w{str(graph_window)}-o{graph_overlap}")
if not os.path.exists(output_path):
    os.makedirs(output_path)


#%% Multiprocessing Functions
def process_videos(video_paths, video_object_locations, graph_window, graph_overlap, selected_class, output_path):

    # Graph Construction
    graph_constructor = GraphConstructor()

    for video_path in video_paths:

        # Get the video name
        video_name = video_path.split(os.sep)[-1]

        # Find the object location for a video path
        object_dir = [key for key in video_object_locations.keys() if video_name.split(".")[0] in
                      video_object_locations[key]][0]

        # Path to objects in each frame of the video
        video_object_full_path = os.path.join(object_dir, video_name.split(".")[0])
        # Path to the video
        video_data_full_path = video_path
        # Path to video annotation
        video_annotation_full_path = os.path.join(*video_path.split(os.sep)[0:-1],
                                                  "human_" + video_name.split(".")[0] + ".csv")
        if os.name == "posix":
            video_annotation_full_path = "/" + video_annotation_full_path

        # Save the output
        save_path = os.path.join(output_path, video_name.split(".")[0] + ".pkl")
        if os.path.exists(save_path):
            print("Skipping " + video_name.split(".")[0])
            continue

        # Construct gdr
        graph_constructor.process_single_video(video_object_full_path, video_data_full_path,
                                               video_annotation_full_path, graph_window, graph_overlap, selected_class)

        with open(save_path, "wb") as fhandle:
            pickle.dump(graph_constructor.data_list, fhandle, protocol=pickle.HIGHEST_PROTOCOL)
        print("DATA SAVED at {}".format(output_path))

    return True


#%% Multiprocessing Initialization
# Get the number of cores to use
num_cores = cfg["multiprocessing"]["num_cores"]
# Load all the paths to the video
video_names = []
for video_path in video_paths:
    video_names.extend([os.path.join(video_path, x) for x in os.listdir(video_path) if x.split(".")[-1] == "mp4"])

# Split the graphs into num_cores segments
video_paths_groups = np.array_split(np.array(video_names), num_cores, axis=0)

# Send data to each core
print("Assigning tasks to different CPUs")
with concurrent.futures.ProcessPoolExecutor() as executor:

    completed_graph_processing = [executor.submit(process_videos, video_paths_group, video_object_locations,
                                                  graph_window, graph_overlap, selected_class, output_path)
                                  for video_paths_group in video_paths_groups]

    # Wait for all CPUs to complete
    for result in concurrent.futures.as_completed(completed_graph_processing):
        result.result()

print("All the CPUs have completed processing the graphs")

