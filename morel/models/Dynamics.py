import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import scipy.spatial
import os
import tarfile

class DynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons,n_layers, activation):
        super(DynamicsNet, self).__init__()

        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, n_neurons),nn.LayerNorm(n_neurons), activation(),
        )
        self.model_list = nn.ModuleList()
        for _ in range(n_layers):
            self.model_list.append(nn.Sequential(
                nn.Linear(n_neurons, n_neurons),nn.LayerNorm(n_neurons), activation()
            ))
        self.output_layer = nn.Linear(n_neurons, output_dim)


    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.model_list:
            x = layer(x)+x
        x = self.output_layer(x)
        
        return  x

class DynamicsEnsemble():
    def __init__(self, input_dim, output_dim, n_models = 4, n_neurons = 256, threshold = 3.0, n_layers = 2, activation = nn.SiLU, cuda = True):
        self.n_models = n_models
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation = activation

        self.threshold = threshold

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.models = []

        for i in range(n_models):
            if(cuda):
                self.models.append(DynamicsNet(input_dim,
                                            output_dim,
                                            n_neurons = n_neurons,
                                            n_layers = n_layers,
                                            activation = activation).cuda())
            else:
                self.models.append(DynamicsNet(input_dim,
                                            output_dim,
                                            n_neurons = n_neurons,
                                            n_layers = n_layers,
                                            activation = activation))

    def forward(self, model, x):
        return self.models[model](x)

    def train_step(self, model_idx, feed, target):
        # Reset Gradients
        self.optimizers[model_idx].zero_grad()

        # Feed forward
        next_state_pred = self.models[model_idx](feed)
        output = self.losses[model_idx](next_state_pred, target)

        # Feed backwards
        output.backward()

        # Weight update
        self.optimizers[model_idx].step()

        # Tensorboard
        return output


    def train(self, dataloader, epochs, loss, summary_writer = None, comet_experiment = None):
        dynamics_lr_start = 1e-4
        dynamics_lr_end = 1e-6
        hyper_params = {
            "dynamics_n_models":  self.n_models,
            "usad_threshold": self.threshold,
            "dynamics_loss_fn" : loss,
            "dynamics_activation" : self.activation,
            "dynamics_epochs" : epochs,
            "dynamics_n_neurons" : self.n_neurons,
            "dynamics_n_layers" : self.n_layers,
            "dynamics_lr_start" : dynamics_lr_start,
            "dynamics_lr_end" : dynamics_lr_end
            
        }
        if(comet_experiment is not None):
            comet_experiment.log_parameters(hyper_params)

        # Define optimizers and loss functions
        self.optimizers = [None] * self.n_models
        self.lr_scheduler = [None] * self.n_models
        self.losses = [None] * self.n_models

        for i in range(self.n_models):
            self.optimizers[i] = torch.optim.Adam(self.models[i].parameters(), lr = dynamics_lr_start)
            self.lr_scheduler[i] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers[i], epochs,dynamics_lr_end)
            self.losses[i] = loss()

        # Start training loop
        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(dataloader)):
                # Split batch into input and output
                feed, target = batch

                loss_vals = list(map(lambda i: self.train_step(i, feed, target), range(self.n_models)))

                # Tensorboard
                if(summary_writer is not None):
                    for j, loss_val in enumerate(loss_vals):
                        summary_writer.add_scalar('Loss/dynamics_{}'.format(j), loss_val, epoch*len(dataloader) + i)

                if(comet_experiment is not None and i % 10 == 0):
                    for j, loss_val in enumerate(loss_vals):
                        comet_experiment.log_metric('dyn_model_{}_loss'.format(j), loss_val, epoch*len(dataloader) + i)
                        comet_experiment.log_metric('dyn_model_avg_loss'.format(j), sum(loss_vals)/len(loss_vals), epoch*len(dataloader) + i)
            for i in range(self.n_models):
                self.lr_scheduler[i].step()
            print("Epoch {} complete\t Last step Loss {}".format(epoch, sum(loss_vals)/len(loss_vals) ))


    def usad(self, predictions):
        # Compute the pairwise distances between all predictions
        distances = scipy.spatial.distance_matrix(predictions, predictions)

        # If maximum is greater than threshold, return true
        return (np.amax(distances) > self.threshold)

    def predict(self, x):
        # Generate prediction of next state using dynamics model
        with torch.set_grad_enabled(False):
            return torch.stack(list(map(lambda i: self.forward(i, x), range(self.n_models))))

    def save(self, save_dir):
        for i in range(self.n_models):
            torch.save(self.models[i].state_dict(), os.path.join(save_dir, "dynamics_{}.pt".format(i)))

    def load(self, load_dir):
        for i in range(self.n_models):
            self.models[i].load_state_dict(torch.load(os.path.join(load_dir, "dynamics_{}.pt".format(i))))

