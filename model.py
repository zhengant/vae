import torch


class VAEEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAEEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.tanh1 = torch.nn.Tanh()
        self.fc_mean = torch.nn.Linear(hidden_size, latent_size)
        self.fc_logvar = torch.nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        hidden = self.tanh1(self.fc1(x))
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)

        return mean, logvar


class VAEDecoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(VAEDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_size, hidden_size)
        self.tanh1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = self.tanh1(self.fc1(x))
        output = self.fc2(hidden)

        return output


class VAE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, device):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = VAEEncoder(input_size, hidden_size, latent_size)
        self.decoder = VAEDecoder(latent_size, hidden_size, input_size)

        self.device = device

    def forward(self, x, num_samples=1):
        batch_size = x.shape[0]
        x = x.flatten(start_dim=1)
        mean, logvar = self.encoder(x)

        # reparameterization trick
        eps = torch.randn((batch_size, num_samples, self.latent_size)).to(self.device)
        z = (
            mean.view((batch_size, 1, self.latent_size))
            + logvar.view((batch_size, 1, self.latent_size)).exp() * eps
        )

        output = self.decoder(z)

        return output, mean, logvar


def vae_loss(x, xhat, mean, logvar):
    x = x.flatten(start_dim=1)
    kl_term = -0.5 * torch.sum((1 + logvar - torch.pow(mean, 2) - logvar.exp()))

    likelihood_term = 0
    bce = torch.nn.BCEWithLogitsLoss()
    num_samples = xhat.shape[1]
    for i in range(num_samples):
        likelihood_term += bce(xhat[:, i, :], x.view(-1, x.shape[1]))

    return kl_term + likelihood_term / num_samples, kl_term, likelihood_term
