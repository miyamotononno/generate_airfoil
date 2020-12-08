import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from models import Generator, Discriminator
import optuna

DEVICE = torch.device("cpu")
b1 = 0.5
b2 = 0.999
FloatTensor = torch.FloatTensor

perfs_npz = np.load("./dataset/standardized_perfs.npz")
coords_npz = np.load("./dataset/standardized_coords.npz")
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]
perfs = perfs_npz[perfs_npz.files[0]]
perf_mean = perfs_npz[perfs_npz.files[1]]
perf_std = perfs_npz[perfs_npz.files[2]]

dataset = torch.utils.data.TensorDataset(torch.tensor(coords), torch.tensor(perfs))

def objective(trial):
    # Generate the model.

    latent_dim = trial.suggest_int("latent_dim", 1, 100)
    G = Generator(latent_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)

    loss_func = trial.suggest_categorical("adversarial_loss", ["BCELoss", "MSELoss"])
    adversarial_loss = getattr(nn, loss_func)()

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
    )

    for _ in range(200):
        for i, (coords, labels) in enumerate(dataloader):
            batch_size = coords.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(coords.type(FloatTensor))
            labels = Variable(torch.reshape(labels.float(), (batch_size, 1)))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))

            gen_labels = Variable(FloatTensor(np.random.normal(loc=perf_mean, scale=perf_std, size=(batch_size, 1))))
            # Generate a batch of images
            gen_imgs = G(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity = D(gen_imgs, gen_labels)
            g_loss = - adversarial_loss(validity, fake)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = D(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = D(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

    return (abs(d_loss.item())+abs(g_loss.item()))/2

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))