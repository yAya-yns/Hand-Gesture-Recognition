from torch import nn

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
