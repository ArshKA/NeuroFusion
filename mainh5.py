from DataLoaderH5 import DatasetH5
from model import BrainModel
import torch
import torch.optim as optim
from tqdm import tqdm

device = 'cuda:3'
n_epochs = 50

mri_data = DatasetH5('/scratch/arsh/mri_h5/scans/mri_data.hdf5', '/scratch/arsh/mri_h5/stimuli/stimuli_data.hdf5', 1, 'D', list(range(1, 16)), device=device, norm_path='norm')

train_size = int(0.8 * len(mri_data))
test_size = len(mri_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(mri_data, [train_size, test_size])

print(test_dataset.dataset.order)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

mri_model = BrainModel()

# mri_model.load_state_dict(torch.load('norm2_dropout_model.pt'))
mri_model.to(device)
print("Transferred model to cuda")


loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(mri_model.parameters(), lr=0.0003)

train_loss = []
test_loss = []

for epoch in range(n_epochs):
    train_loss.append([])
    for X_batch, y_batch in (pbar := tqdm(train_generator, colour='green')):
        y_pred = mri_model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        pbar.set_description(f"Loss {loss}")
        train_loss[-1].append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        epoch_test_losses = []
        for X_batch, y_batch in test_generator:
            y_pred = mri_model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            epoch_test_losses.append(float(loss))
        test_loss.append(sum(epoch_test_losses)/len(epoch_test_losses))
        if test_loss[-1] == min(test_loss):
            torch.save(mri_model.state_dict(), f'models/norm5_epoch{epoch}_{test_loss[-1]:.3f}.pt')
            print(f"Saved Epoch {epoch}, Loss: {test_loss[-1]:.3f}")
        print(f"Epoch {epoch}, Loss {sum(train_loss[-1])/len(train_loss[-1])}, Test Loss {test_loss[-1]}")

torch.save(mri_model.state_dict(), 'norm5_dropout_model.pt')

