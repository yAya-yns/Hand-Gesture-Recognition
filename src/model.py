from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=16, layer_two_neurons=128, layer_three_neurons=64):

        self.layer_two_neurons = layer_two_neurons
        self.layer_three_neurons = layer_three_neurons
        self.latent_dim = latent_dim
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(42, self.layer_two_neurons),
            nn.ReLU(),
            nn.Linear(self.layer_two_neurons, self.layer_three_neurons),
            nn.ReLU(),
            nn.Linear(self.layer_three_neurons, self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.layer_three_neurons),
            nn.ReLU(),
            nn.Linear(layer_three_neurons, layer_two_neurons),
            nn.ReLU(),
            nn.Linear(self.layer_two_neurons, 42),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_layer_dim=16, layer_two_dim=16, num_of_output_class=11):
    
        self.input_layer_dim = input_layer_dim
        self.layer_two_dim = layer_two_dim
        self.num_of_output_class = num_of_output_class
        
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(self.input_layer_dim, self.layer_two_dim)
        self.fc2 = nn.Linear(self.layer_two_dim, self.num_of_output_class)
        
    def forward(self, x):
        x = x.view(-1, self.input_layer_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x