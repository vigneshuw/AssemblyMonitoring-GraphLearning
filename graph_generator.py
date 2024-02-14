#%% Imports
import os
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

# List all the videos inside the object paths
video_object_locations = {}
for video_object_path in video_object_paths:
    video_object_locations[video_object_path] = os.listdir(video_object_path)

# Output path for saving the processed videos
output_path = os.path.join(cfg['output_path'], f"w{str(graph_window)}-o{graph_overlap}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

#%% Graph Construction
graph_constructor = GraphConstructor()

for video_path in video_paths:
    # List all the videos inside
    video_names = [x for x in os.listdir(video_path) if x.split('.')[-1] == 'mp4']

    # Create graphs for every video that is inside
    for video_name in video_names:
        # Find the object location for the video name
        object_dir = [key for key in video_object_locations.keys() if video_name.split(".")[0] in
                      video_object_locations[key]][0]

        # The path to the objects in each frame for a video
        video_object_full_path = os.path.join(object_dir, video_name.split(".")[0])
        # Path to the video
        video_data_full_path = os.path.join(video_path, video_name)
        # Path to video annotation
        video_annotation_full_path = os.path.join(video_path, "human_" + video_name.split(".")[0] + ".csv")

        # Construct graphs
        graph_constructor.process_single_video(video_object_full_path, video_data_full_path,
                                               video_annotation_full_path, graph_window, graph_overlap)

        # Save the output
        save_path = os.path.join(output_path, video_name.split(".")[0] + ".pkl")

        with open(save_path, "wb") as fhandle:
            pickle.dump(graph_constructor.data_list,  fhandle, protocol=pickle.HIGHEST_PROTOCOL)
        print("DATA SAVED at {}".format(output_path))
