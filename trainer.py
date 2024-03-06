import os
import random
import torch
import sys
import signal
import datetime
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


class TrainingDataGenerator:

    def __init__(self, proportion_valid: float, proportion_test: float, batch_size: int):
        # Initialization
        self.proportion_valid = proportion_valid
        self.proportion_test = proportion_test
        self.batch_size = batch_size
        # Datasets
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        # DataLoaders
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        # Labels
        self.train_labels = None
        self.valid_labels = None
        self.test_labels = None
        self.unique_labels = None

        # Weights
        self.class_weights = None

    def generate_data(self, data_items):
        # Shuffle the data
        random.shuffle(data_items)

        # Print the available labels
        self.unique_labels = set()
        for data in data_items:
            self.unique_labels.add(data["frame_window"].y.item())
        self.unique_labels = list(self.unique_labels)
        print("The list of available labels: ", end="")
        print(self.unique_labels)

        # Training dataset
        self.train_dataset, self.valid_dataset = train_test_split(data_items, test_size=self.proportion_valid,
                                                                  random_state=42)
        self.valid_dataset, self.test_dataset = train_test_split(self.valid_dataset, test_size=self.proportion_test,
                                                                 random_state=42)
        print(f'Number of training graphs: {len(self.train_dataset)}')
        print(f'Number of validation graphs: {len(self.valid_dataset)}')
        print(f'Number of testing graphs: {len(self.test_dataset)}')

    def initiate_dataloaders(self, num_workers=1):
        # Generate DataLoaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True,
                                           shuffle=True, num_workers=num_workers)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, pin_memory=True,
                                           shuffle=True, num_workers=num_workers)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True,
                                          num_workers=num_workers)

        # Get the label
        self.train_labels = torch.tensor([data['frame_window'].y for data in self.train_dataset])
        self.valid_labels = torch.tensor([data['frame_window'].y for data in self.valid_dataset])
        self.test_labels = torch.tensor([data['frame_window'].y for data in self.test_dataset])

        # Class weights
        class_counts = torch.bincount(self.train_labels)
        total_counts = class_counts.sum().item()
        self.class_weights = 1.0 - (class_counts / total_counts)
        print("Class weights: ", self.class_weights)


class TestingDataGenerator:

    def __init__(self, batch_size: int):

        # Initialize
        self.batch_size = batch_size
        self.test_dataset = None
        self.test_dataloader = None
        self.test_labels = None
        self.unique_labels = None

    def generate_data(self, data_items, shuffle=False):

        if shuffle:
            random.shuffle(data_items)

        # Print the available labels
        self.unique_labels = set()
        for data in data_items:
            self.unique_labels.add(data["frame_window"].y.item())
        self.unique_labels = list(self.unique_labels)
        print("The list of available labels: ", end="")
        print(self.unique_labels)

        self.test_dataset = data_items
        print(f'Number of testing graphs: {len(self.test_dataset)}')

    def initiate_dataloaders(self, shuffle=False):

        # Generate DataLoaders
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True,
                                          shuffle=shuffle)
        self.test_labels = torch.tensor([data['frame_window'].y for data in self.test_dataset])


class Trainer:

    def __init__(self, model, optimizer, loss_fn, dataloaders: tuple, gpu_ids: tuple = (3,), es_patience=30):

        # Select the GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)

        # Initialize
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader, self.valid_dataloader, self.test_dataloaders = dataloaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        if torch.cuda.is_available():
            print("Training on GPU using Distributed DataParallel")
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # Check the weights on loss_fn
        if self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(device=self.device)
        # Training state
        self.stop_train = False
        signal.signal(signal.SIGINT, self.stop_training)

        # Early Stopping
        if es_patience is not None and es_patience > 0:
            self.early_stopper = EarlyStopper(patience=es_patience)
            sys.stdout.write("Early Stopping enabled. \n")
        else:
            self.early_stopper = None
            sys.stdout.write("Early Stopping disabled. \n")

        # Training history information
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
            "all_predictions": [],
            "all_labels": [],
        }

    def train(self, epochs):

        # Set the model for training
        self.model.train()

        for epoch in range(epochs + 1):
            all_preds = []
            all_labels = []
            loss_train = 0.0
            correct_train = 0
            total = 0
            total_batch = 0
            # Reset to training
            self.model.train()
            for batch_id, data in enumerate(tqdm(self.train_dataloader)):
                # Forward
                data = data.to(self.device)
                output = self.model(data)
                with torch.no_grad():
                    pred = output.argmax(dim=1)

                # Backward
                loss = self.loss_fn(output, data['frame_window'].y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()
                correct_train += int((pred == data['frame_window'].y).sum())
                total += len(data)
                total_batch += 1

            # Evaluation
            total_val, total_val_batch, loss_val, correct_val, all_labels_epoch, all_preds_epoch = self.validate_model()
            # Record the values
            all_labels.extend(all_labels_epoch)
            all_preds.extend(all_preds_epoch)
            self.history["train_losses"].append(loss_train / total_batch)
            self.history["val_losses"].append(loss_val / total_val_batch)
            self.history["train_acc"].append(correct_train / total)
            self.history["val_acc"].append(correct_val / total_val)
            print(
                '{} Epoch-{}, train-loss {:.4f}, train-acc {:.4f} || val-loss {:.4f}, val-acc {:.4f} || lr - {}'.format(
                    datetime.datetime.now(), epoch, loss_train / total_batch,
                                                    correct_train / total, loss_val / total_val_batch,
                                                    correct_val / total_val,
                    self.optimizer.param_groups[0]['lr']))

            # Check for early stopping or training stopping
            if self.early_stopper is not None:
                if self.early_stopper.early_stop(loss_val):
                    break
            if self.stop_train:
                break

    def stop_training(self, sig, frame):

        sys.stdout.write("\n\n" + "=" * 40)
        sys.stdout.write("\nStopping training after this epoch\n")
        self.stop_train = True

    def validate_model(self):

        # Model in eval mode
        self.model.eval()

        # Params
        all_preds_epoch = []
        all_labels_epoch = []
        correct = 0
        total = 0
        total_batch = 0
        loss_val = 0.0

        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(self.valid_dataloader)):
                # Forward
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)

                total += len(data)
                total_batch += 1
                correct += int((pred == data['frame_window'].y).sum())
                loss = self.loss_fn(output, data['frame_window'].y)
                loss_val += loss.item()
                all_preds_epoch.extend(pred.cpu().numpy())
                all_labels_epoch.extend(data['frame_window'].y.cpu().numpy())

        return total, total_batch, loss_val, correct, all_labels_epoch, all_preds_epoch

    def test_model(self, test_loader):

        # Model in eval mode
        self.model.eval()

        # Params
        all_preds_epoch = []
        all_labels_epoch = []
        correct = 0
        total = 0
        total_batch = 0
        loss_test = 0.0

        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(test_loader)):

                # Forward
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)

                total += len(data)
                total_batch += 1
                correct += int((pred == data['frame_window'].y).sum())
                loss = self.loss_fn(output, data['frame_window'].y)
                loss_test += loss.item()
                all_preds_epoch.extend(pred.cpu().numpy())
                all_labels_epoch.extend(data['frame_window'].y.cpu().numpy())

        return total, total_batch, loss_test, correct, all_labels_epoch, all_preds_epoch


class EarlyStopper:
    """
    The EarlyStopper class for training
    """

    def __init__(self, patience=30, min_delta=0.015):
        """
        Initialization

        :param patience: Number of epochs to wait
        :param min_delta: The minimum change that is required
        """

        # Init
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """
        Early stop instance

        :param validation_loss: The loss to monitor
        :return:
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                sys.stdout.write("EarlyStopping the training process due to non changing validation loss\n")
                return True
        return False