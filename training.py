#%%
import os
import yaml
from graph_construction import GraphConstructor
import trainer
import torch
import matplotlib.pyplot as plt
from models.gcn import RGCN
from sklearn.metrics import f1_score, classification_report

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
plt.plot(range(1, len(model_trainer.history["train_losses"]) + 1), model_trainer.history['train_losses'],
         label='Train Loss', color='red')
plt.plot(range(1, len(model_trainer.history["val_losses"]) + 1), model_trainer.history['val_losses'],
         label='Validation Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(range(1, len(model_trainer.history["train_acc"]) + 1), model_trainer.history['train_acc'],
         label='Train Accuracy', color='red')
plt.plot(range(1, len(model_trainer.history["val_acc"]) + 1), model_trainer.history['val_acc'],
         label='Validation Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% Testing trained model
print("Testing the trained model")

# Initiate the data params
graph_constructor_test = GraphConstructor(data_params=yaml_file_params, assembly=assembly)

# Load annotation and construct graphs
graph_constructor_test.load_data_and_labels(data_type="testing")
processed_objects_path = yaml_file_params[assembly]["processed_objects_information"]["testing"]["path"]
window_size = yaml_file_params[assembly]["processed_objects_information"]["testing"]["window_size"]
overlap = yaml_file_params[assembly]["processed_objects_information"]["testing"]["overlap"]
graph_constructor_test.construct_graphs(processed_objects_path, window_size=window_size, overlap=overlap)

# Create DataLoaders
test_data_generator = trainer.TestingDataGenerator(batch_size=128)
test_data_generator.generate_data(graph_constructor.data_list)
test_data_generator.initiate_dataloaders()

# Test on a new data
result = model_trainer.test_model(test_data_generator.test_dataloader)

#%% Evaluation parameters
test_f1_score = f1_score(result[-2], result[-1], average="weighted")
print(f"F1-Score across the testing data is {test_f1_score}")
test_classification_report = classification_report(result[-2], result[-1])
print(test_classification_report)

#%% Save the model
save_dir = os.path.join(os.getcwd(), "trained_models")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

items_to_save = {
    "history": model_trainer.history,
    "model_state_dict": model.state_dict(),
    "test_results": result
}
torch.save(items_to_save, os.path.join(save_dir, "gcnn_L10.pt"))
