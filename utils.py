import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def make_ckpt_data_sample_dirs():
    os.makedirs("checkpoints/", exist_ok=True)
    os.makedirs("data/", exist_ok=True)
    os.makedirs("samples/", exist_ok=True)


def save_checkpoint(model, epoch):
    filename = "vae_ckpt_epoch" + str(epoch) + ".pt"
    filepath = os.path.join("checkpoints", filename)

    torch.save(model.state_dict(), filepath)


def sample_from_prior(model, batch_size, device):
    z = torch.randn((batch_size, model.latent_size)).to(device)

    model.eval()
    output = model.decoder(z)
    output = torch.sigmoid(output)
    samples = output.data.cpu().numpy()

    return samples


def plot_samples_from_prior(model, device, epoch, image_dims, cmap="Greys_r"):
    samples = sample_from_prior(model, 10, device)

    fig = plt.figure(figsize=(10, 1))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")

        plt.imshow(sample.reshape(image_dims), cmap=cmap)

    save_path = os.path.join("samples", "from_prior" + str(epoch) + ".png")
    plt.savefig(save_path)
    plt.close(fig)


def plot_reconstructions(model, batch, device, epoch, image_dims, cmap="Greys_r"):
    x = batch[0]
    x = x.to(device)
    output, _, _ = model(x)

    originals = x.data.cpu().numpy()
    reconstructions = output.data.cpu().numpy()

    fig, axes = plt.subplots(5, 2)
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Reconstructed")
    for i in range(5):
        axes[i, 0].imshow(originals[i].reshape(image_dims), cmap=cmap)
        axes[i, 1].imshow(reconstructions[i].reshape(image_dims), cmap=cmap)
        axes[i, 0].set_xticklabels([])
        axes[i, 1].set_yticklabels([])

    save_path = os.path.join("samples", "reconstructions" + str(epoch) + ".png")
    plt.savefig(save_path)
    plt.close(fig)
