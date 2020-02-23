import numpy as np
import torch
from torchvision import datasets, transforms


def load_mnist(batch_size=64):
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(
        "data/", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        "data/", train=False, download=True, transform=transform
    )
    n = len(mnist_train)
    val_len = int(0.2 * n)
    mnist_train, mnist_val = torch.utils.data.random_split(
        mnist_train, [n - val_len, val_len]
    )

    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        mnist_val, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader, test_loader


def load_centered_dspirtes(n=60000, batch_size=64):
    dataset_zip = np.load(
        "data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        allow_pickle=True,
        encoding="latin1",
    )
    imgs = dataset_zip["imgs"]
    latents_values = dataset_zip["latents_values"]
    latents_classes = dataset_zip["latents_classes"]
    metadata = dataset_zip["metadata"][()]
    latents_possible_values = metadata["latents_possible_values"]
    latents_sizes = metadata["latents_sizes"]
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
    )

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def show_images_grid(imgs_, num_images=25):
        ncols = int(np.ceil(num_images ** 0.5))
        nrows = int(np.ceil(num_images / ncols))
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i < num_images:
                ax.imshow(imgs_[ax_i], cmap="Greys_r", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
        plt.show()

    # find center value
    # center_x = latents_possible_values["posX"][int(latents_sizes[-2] / 2)]
    # center_y = latents_possible_values["posY"][int(latents_sizes[-1] / 2)]
    center_value = 0.51612903

    ## Fix posX and posY latent to center
    latents_sampled = sample_latent(size=n)
    latents_sampled[:, -2] = center_value
    latents_sampled[:, -1] = center_value
    indices_sampled = latent_to_index(latents_sampled)
    imgs_sampled = imgs[indices_sampled]
    imgs_sampled = imgs_sampled.astype(np.float32)

    show_images_grid(imgs_sampled)

    # split out validation and testing
    training, testing = train_test_split(imgs_sampled, test_size=0.2)
    training, valid = train_test_split(training, test_size=0.2)

    # dummy labels to make iteration the same as with mnist
    train_labels = np.empty((training.shape[0], 1))
    valid_labels = np.empty((valid.shape[0], 1))
    test_labels = np.empty((testing.shape[0], 1))

    # make tensors
    training = torch.from_numpy(training)
    valid = torch.from_numpy(valid)
    testing = torch.from_numpy(testing)
    train_labels = torch.from_numpy(train_labels)
    valid_labels = torch.from_numpy(valid_labels)
    test_labels = torch.from_numpy(test_labels)

    # torch datasets
    train_dataset = torch.utils.data.TensorDataset(training, train_labels)
    valid_dataset = torch.utils.data.TensorDataset(valid, valid_labels)
    test_dataset = torch.utils.data.TensorDataset(testing, test_labels)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader, test_loader
