#%%
import argparse
import os
import yaml
from graph_construction import GraphConstructor
import trainer
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from models.gcn import RGCN
from sklearn.metrics import f1_score, classification_report


#%% Functions for saving data and plotting
def plot_train_results(model_trainer, save_dir):

    plt.plot(range(1, len(model_trainer.history["train_losses"]) + 1), model_trainer.history['train_losses'],
             label='Train Loss', color='red')
    plt.plot(range(1, len(model_trainer.history["val_losses"]) + 1), model_trainer.history['val_losses'],
             label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    save_fname = 'train-loss.png'
    plt.savefig(os.path.join(save_dir, save_fname))
    plt.clf()

    plt.plot(range(1, len(model_trainer.history["train_acc"]) + 1), model_trainer.history['train_acc'],
             label='Train Accuracy', color='red')
    plt.plot(range(1, len(model_trainer.history["val_acc"]) + 1), model_trainer.history['val_acc'],
             label='Validation Accuracy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    save_fname = 'train-acc.png'
    plt.savefig(os.path.join(save_dir, save_fname))
    plt.clf()


def evaluate(model_trainer, dataloader, type, save_dir):

    result = model_trainer.test_model(dataloader)
    test_f1_score = f1_score(result[-2], result[-1], average="weighted")
    print(f"F1-Score across the testing data is {test_f1_score}")
    test_classification_report = classification_report(result[-2], result[-1], output_dict=True)
    report_df = pd.DataFrame(test_classification_report).transpose()

    # Save the DataFrame to a CSV file
    save_file = f"classification_report-{type}.csv"
    report_df.to_csv(os.path.join(save_dir, save_file), index=True)

    # Save the actual predictions and labels
    conf_mat = {
        "labels": result[-2],
        "predictions": result[-1],
    }
    save_file = F"confusion-matrix-{type}.pkl"
    pickle.dump(conf_mat, open(os.path.join(save_dir, save_file), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("All information saved successfully!")

#%% Construct Graphs
# Open the YAML file
with open("training.yml", "r") as filehandle:
    yaml_file_params = yaml.load(filehandle, Loader=yaml.FullLoader)

# Initiate the data params
assembly = yaml_file_params["assembly_selected"]
graph_constructor = GraphConstructor(data_params=yaml_file_params, assembly=assembly)

# Load annotation
graph_constructor.load_data_and_labels()
processed_objects_path = yaml_file_params[assembly]["processed_objects_information"]["training"]["path"]
window_size = yaml_file_params[assembly]["processed_objects_information"]["training"]["window_size"]
overlap = yaml_file_params[assembly]["processed_objects_information"]["training"]["overlap"]
selected_class = yaml_file_params[assembly]["processed_objects_information"]["training"]["selected_classes"]
processed_graphs = yaml_file_params[assembly]["processed_graphs"]
temp_sc = [str(x) for x in selected_class]
save_dir = os.path.join(yaml_file_params[assembly]["save_dir"], "sc-" + "-".join(temp_sc),
                        f"w{window_size}-o{overlap}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(f"All the models will be saved at -> {save_dir}")

# Videos directory
training_video_dir = yaml_file_params[assembly]["data_dir"]["training"]
testing_video_dir = yaml_file_params[assembly]["data_dir"]["testing"]

if len(selected_class) > 1:
    class_save_dir = "sc-" + "-".join([str(x) for x in selected_class])
else:
    class_save_dir = "sc-" + str(selected_class[0])

# Construct if not available
if processed_graphs:
    print("Graphs exist! Skipping graph construction...")
    # Load the data appropriately
    processed_graphs = os.path.join(processed_graphs, class_save_dir, f"w{window_size}-o{overlap}")

    # Training generator
    training_graph_paths = [os.path.join(processed_graphs, x.split(".")[0] + ".pkl") for x in
                            os.listdir(training_video_dir) if x.split(".")[-1] == "mp4"]
    # Testing data list
    testing_graph_paths = [os.path.join(processed_graphs, x.split(".")[0] + ".pkl") for x in
                           os.listdir(testing_video_dir) if x.split(".")[-1] == "mp4"]

    # Load the data
    print("Loading created graphs...")
    training_data = []
    for graph_path in training_graph_paths:
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        training_data.extend(data)
    print(f"Total number of training graphs are {len(training_data)}")

else:
    print("Constructing graphs...")
    raise NotImplementedError
    # graph_constructor.construct_graphs(processed_objects_path, window_size=window_size, overlap=overlap,
    #                                    selected_class=selected_class)

#%% Create DataLoaders

data_generator = trainer.TrainingDataGenerator(proportion_valid=0.3, proportion_test=0.5, batch_size=256)
data_generator.generate_data(training_data)
data_generator.initiate_dataloaders(num_workers=4)

#%% Construct model
in_channels = training_data[0].num_features["frame_window"]
hidden_channels = 64
num_layers = 2
output_channels = len(data_generator.unique_labels)
model = RGCN(in_channels, hidden_channels, output_channels, num_layers)
# Start the training process
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss(weight=data_generator.class_weights)
model_trainer = trainer.Trainer(
    model, optimizer, loss_fn,
    (data_generator.train_dataloader, data_generator.valid_dataloader, data_generator.test_dataloader),
    gpu_ids=(3, ))
# Train
model_trainer.train(epochs=200)
plot_train_results(model_trainer, save_dir)
# Integrated test
evaluate(model_trainer, data_generator.test_dataloader, type="integratedTest", save_dir=save_dir)

#%% Testing trained model
testing_data = []
for graph_path in testing_graph_paths:
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    testing_data.extend(data)
print(f"Total number of testing graphs are {len(testing_data)}")

# Evaluate on the new unseen test data
test_data_generator = trainer.TestingDataGenerator(batch_size=256)
test_data_generator.generate_data(testing_data)
test_data_generator.initiate_dataloaders()
evaluate(model_trainer, test_data_generator.test_dataloader, type="unseenTest", save_dir=save_dir)

#%% Save the trained model
model_save_dict = {
    "model": {
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "output_channels": output_channels,
        "num_layers": num_layers}
}
torch.save(model_save_dict, os.path.join(save_dir, "gcn.pt"))
