#%%
import os
import yaml
from graph_construction import GraphConstructor
import trainer
import torch
from models.gcn import RGCN
from importlib import reload
reload(trainer)

#%% Construct Graphs
# Open the YAML file
with open("data.yml", "r") as filehandle:
    yaml_file_params = yaml.load(filehandle, Loader=yaml.FullLoader)

# Initiate the data params
assembly = "L10"
graph_constructor = GraphConstructor(data_params=yaml_file_params, assembly=assembly)

# Load annotation
graph_constructor.load_data_and_labels()
processed_objects_path = yaml_file_params[assembly]["processed_objects_information"]["training"]["path"]
window_size = yaml_file_params[assembly]["processed_objects_information"]["training"]["window_size"]
overlap = yaml_file_params[assembly]["processed_objects_information"]["training"]["overlap"]
graph_constructor.construct_graphs(processed_objects_path, window_size=window_size, overlap=overlap)

#%% Create DataLoaders
data_generator = trainer.TrainingDataGenerator(proportion_valid=0.3, proportion_test=0.5, batch_size=128)
data_generator.generate_data(graph_constructor.data_list)
data_generator.initiate_dataloaders()

#%% Construct Models
in_channels = graph_constructor.data_list[0].num_features["frame_window"]
hidden_channels = 64
num_layers = 2
output_channels = len(data_generator.unique_labels)
print(f"Number of input channels: {in_channels}, Number of output channels: {output_channels}")
model = RGCN(in_channels, hidden_channels, output_channels, num_layers)
print(model)

#%% Start the training process
# Training params
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_fn = torch.nn.CrossEntropyLoss(weight=data_generator.class_weights)
model_trainer = trainer.Trainer(
    model, optimizer, loss_fn,
    (data_generator.train_dataloader, data_generator.valid_dataloader, data_generator.test_dataloader))
# Train
model_trainer.train(epochs=500)
#%% Plotting the results


