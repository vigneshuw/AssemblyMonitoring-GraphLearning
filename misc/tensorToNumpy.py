#%%
import pickle
import os
import glob
from tqdm import tqdm

#%% Load the files for conversion
directory_name = "/data1/GraphModellingExperiments/L10/ObjectDetections/C3D/WithoutHands/RPN_Train_Results-UnreasonablyLarge"
pickle_files = glob.glob(directory_name + "/**/*.pickle")


#%% Read files one by one and convert to NumPy
for file in tqdm(pickle_files):
    with open(file, "rb") as f:
        data = pickle.load(f)

    # Convert the features to numpy
    data["features"] = data["features"].detach().cpu().numpy()

    # Save the file
    temp = file.split(os.sep)
    temp[-3] = temp[-3].split("-")[0]
    save_directory = os.sep + os.path.join(*temp[0:-1])
    save_full_path = os.path.join(save_directory, temp[-1])

    # Create directory if not exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(save_full_path, "wb") as f:
        pickle.dump(data, f)


#%% Load to check the file
path = "/data1/GraphModellingExperiments/L10/ObjectDetections/C3D/WithoutHands/RPN_Test_Results/WIN_20220131_10_14_12_Pro_Jan31_cycle3/1115.pickle"
with open(path, "rb") as f:
    data = pickle.load(f)

print(f"Total Number of classes: {len(data['class'])}\nFeature shape: {data['features'].shape}")
