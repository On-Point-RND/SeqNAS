import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys

sys.path.append("../")

from uqmodule.fc_resnet import FCResNet
from uqmodule.uq_constructor import DUE

from trainers.trainer import Trainer

sns.set()
sns.set_palette("colorblind")


np.random.seed(0)
torch.manual_seed(0)

DEVICE = 1


def make_data(n_samples, noise=0.05, seed=2):
    # make some random sines & cosines
    np.random.seed(seed)
    n_samples = int(n_samples)

    W = np.random.randn(30, 1)
    b = np.random.rand(30, 1) * 2 * np.pi

    x = 5 * np.sign(np.random.randn(n_samples)) + np.random.randn(n_samples).clip(-2, 2)
    y = np.cos(W * x + b).sum(0) + noise * np.random.randn(n_samples)
    return x[..., None], y


def get_datasets(n_samples, batch_size=128):
    batch_size = 128
    X_train, y_train = make_data(n_samples)
    X_test, y_test = X_train, y_train
    ds_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=True
    )

    ds_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    )
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=False)

    return ds_train, dl_train, ds_test, dl_test


if __name__ == "__main__":
    n_samples = 1e3
    domain = 15

    x, y = make_data(n_samples)
    plt.scatter(x, y)
    plt.savefig("original.png")

    ds_train, dl_train, ds_test, dl_test = get_datasets(n_samples, batch_size=128)

    steps = 5e3
    epochs = int(steps // len(dl_train) + 1)
    print(f"Training with {n_samples} datapoints for {epochs} epochs")

    input_dim = 1
    features = 128
    depth = 4
    num_outputs = 1  # regression with 1D output
    spectral_normalization = True
    coeff = 0.95
    n_power_iterations = 1
    dropout_rate = 0.01

    n_inducing_points = 10
    kernel = "RBF"

    due = DUE(
        problem_type="reg",  # reg or cls (regression or classification)
        n_inducing_points=10,  # a  number of points in
        kernel="RBF",  # GP kernel
        num_outputs=1,  # num classes or 1 for eregression
        dataset_size=len(ds_train),
        batch_size=128,
        device=DEVICE,
    )

    feature_extractor = FCResNet(
        input_dim=input_dim,
        features=features,
        depth=depth,
        spectral_normalization=spectral_normalization,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
        dropout_rate=dropout_rate,
    )

    model, loss_fn, likelihood = due.make_model_and_loss(feature_extractor, dl_train)

    def criterion(outputs, batch):
        return loss_fn(outputs, batch["target"])

    model.to(DEVICE)

    optimizer = {
        "main": (
            "ADAM",
            {
                "lr": 1e-2,
                "weight_decay": 1e-5,
            },
        )
    }

    trainer = Trainer(
        criterion,
        optimizers=optimizer,
        phases=["train", "validation"],
        num_epochs=epochs,
        device=DEVICE,
        logger=None,
    )

    loaders = {"train": dl_train, "validation": dl_test}
    trainer.set_model(model, {"main": model.parameters()})
    trainer.set_dataloaders(loaders)
    trainer.train()

    model.eval()
    likelihood.eval()

    x_lin = np.linspace(-domain, domain, 100)

    xx = torch.tensor(x_lin[..., None]).float()
    xx = xx.to(DEVICE)
    output, output_std = due.inference(xx, model)
    output, output_std = output.cpu().numpy(), output_std.cpu().numpy()

    plt.xlim(-domain, domain)
    plt.ylim(-10, 10)
    plt.fill_between(
        x_lin, output - output_std, output + output_std, alpha=0.2, color="b"
    )
    plt.fill_between(
        x_lin,
        output - 2 * output_std,
        output + 2 * output_std,
        alpha=0.2,
        color="b",
    )

    plt.scatter([], [])
    plt.scatter([], [])
    X_vis, y_vis = make_data(n_samples=300)

    plt.scatter(X_vis.squeeze(), y_vis, facecolors="none", edgecolors="g", linewidth=2)
    plt.plot(x_lin, output, alpha=0.5)
    plt.savefig("result.png")
