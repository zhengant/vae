import os
import sys
import yaml

import torch
from tqdm import tqdm
import numpy as np

from data import load_mnist, load_centered_dspirtes
from model import VAE, vae_loss
from utils import (
    save_checkpoint,
    plot_samples_from_prior,
    plot_reconstructions,
    make_ckpt_data_sample_dirs,
)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    device,
    image_dims,
    save_freq=5,
    data_cmap="Greys_r",
):
    print("Training model...")

    for i in range(epochs):
        model.train()
        tqdm_loader = tqdm(train_loader, desc="Epoch " + str(i))
        running_loss = 0
        total_images = 0
        for (x, _) in tqdm_loader:
            x = x.to(device)
            optimizer.zero_grad()
            output, mean, logvar = model(x)
            loss, _, _ = vae_loss(x, output, mean, logvar)
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            running_loss += loss.item() * batch_size
            total_images += batch_size

            tqdm_loader.set_postfix({"training_loss": running_loss / total_images})

        model.eval()
        val_loss = test_model(model, val_loader, device)
        print("\tValidation loss: " + str(val_loss))

        if i % save_freq == 0:
            save_checkpoint(model, i)
            plot_reconstructions(
                model, next(iter(val_loader)), device, i, image_dims, data_cmap
            )
            plot_samples_from_prior(model, device, i, image_dims, data_cmap)


def test_model(model, test_loader, device):
    model.eval()
    for (x, _) in test_loader:
        x = x.to(device)
        output, mean, logvar = model(x)
        loss, _, _ = vae_loss(x, output, mean, logvar)

    return loss.item()


def main():
    make_ckpt_data_sample_dirs()

    config_path = sys.argv[1]
    with open(config_path, "r") as config_fp:
        config = yaml.full_load(config_fp)

    input_dims = config["input_dims"]
    input_size = np.prod(input_dims)
    hidden_size = config["hidden_size"]
    latent_size = config["latent_size"]
    device = torch.device(config["device"])
    num_epochs = config["epochs"]
    save_freq = config["save_freq"]

    if config["dataset"] == "mnist":
        train_loader, val_loader, test_loader = load_mnist()
        data_cmap = "Greys_r"
    else:
        train_loader, val_loader, test_loader = load_centered_dspirtes()
        data_cmap = "RGB"

    model = VAE(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        optimizer,
        device,
        image_dims=input_dims,
        save_freq=save_freq,
        data_cmap=data_cmap,
    )
    test_loss = test_model(model, test_loader, device)

    print("Test loss: " + str(test_loss))

    plot_reconstructions(
        model, next(iter(test_loader)), device, num_epochs, input_dims, data_cmap
    )
    plot_samples_from_prior(model, device, num_epochs, input_dims, data_cmap)
    save_checkpoint(model, num_epochs)


if __name__ == "__main__":
    main()
