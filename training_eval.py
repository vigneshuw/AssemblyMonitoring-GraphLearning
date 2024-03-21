#%%
import os
import pickle
import sys

import yaml
import glob
import trainer
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from models.gcn import RGCN


#%% Load Graphs
# Load the YAML Params
with open('training_eval.yml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

processed_graphs = cfg['graph']['processed_graphs']
graph_window = cfg['graph']['window']
graph_overlap = cfg['graph']['overlap']
# Class for temporal information
selected_class = cfg['graph']['selected_class']

if len(selected_class) > 1:
    class_save_dir = "sc-" + "-".join([str(x) for x in selected_class])
else:
    class_save_dir = "sc-" + str(selected_class[0])

# Identify graph names
graph_dir = os.path.join(processed_graphs, cfg['graph']['type'], f"{class_save_dir}",
                         f"w{graph_window}-o{graph_overlap}")
graph_paths = glob.glob(graph_dir + '/*.pkl')
if len(graph_paths) == 0:
    graph_paths = glob.glob(graph_dir + "/*")
    if len(graph_paths) == 0:
        sys.stdout.write("Cannot find processed graphs. Exiting...")

print("Total number of available videos: ", len(graph_paths))

# Configure data save information
training_results_output_dir = os.path.join(cfg["training"]["output"], cfg["graph"]["type"], f"{class_save_dir}",
                                           f"w{graph_window}-o{graph_overlap}")
if not os.path.exists(training_results_output_dir):
    os.makedirs(training_results_output_dir)


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


#%% Load the graphs and train model
split_proportions = cfg['training']['split_props']
for proportion in split_proportions:

    print("Proportion: ", proportion)
    # Check if directory exists - TO save training results
    save_dir = os.path.join(training_results_output_dir, f"test-proportion-{str(proportion)}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(f"Trained results exist for test-proportion-{str(proportion)}")
        continue

    # Get the testing and training videos
    training_graph_paths, testing_graph_paths = train_test_split(graph_paths, test_size=proportion / 100)
    print(f"Number of training graphs: {len(training_graph_paths)} and testing graphs: {len(testing_graph_paths)}")

    # Load the training and testing graphs
    training_data = []
    for graph_path in training_graph_paths:
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        training_data.extend(data)
    testing_data = []
    for graph_path in testing_graph_paths:
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        testing_data.extend(data)
    print(f"Total number of training and testing graphs are {len(training_data)}, and {len(testing_data)} "
          f"respectively")

    # Create data loaders
    data_generator = trainer.TrainingDataGenerator(proportion_valid=0.3, proportion_test=0.5,
                                                   batch_size=cfg['training']['batch_size'])
    data_generator.generate_data(training_data)
    data_generator.initiate_dataloaders(num_workers=4)

    # Construct model
    in_channels = training_data[0].num_features["frame_window"]
    hidden_channels = 64
    num_layers = 2
    output_channels = len(data_generator.unique_labels)
    model = RGCN(in_channels, hidden_channels, output_channels, num_layers)

    # Start the training process
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    loss_fn = torch.nn.CrossEntropyLoss(weight=data_generator.class_weights)
    model_trainer = trainer.Trainer(
        model, optimizer, loss_fn,
        (data_generator.train_dataloader, data_generator.valid_dataloader, data_generator.test_dataloader),
        gpu_ids=(3, ))
    # Train
    model_trainer.train(epochs=500)

    plot_train_results(model_trainer, save_dir)
    evaluate(model_trainer, data_generator.test_dataloader, type="integratedTest", save_dir=save_dir)

    # Evaluate on the new unseen test data
    test_data_generator = trainer.TestingDataGenerator(batch_size=128)
    test_data_generator.generate_data(testing_data)
    test_data_generator.initiate_dataloaders()
    evaluate(model_trainer, test_data_generator.test_dataloader, type="unseenTest", save_dir=save_dir)

    print("=" * 40 + "COMPLETE" + "=" * 40)

