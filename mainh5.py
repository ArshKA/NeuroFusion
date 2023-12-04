from DataLoaderH5 import DatasetH5
from model import BrainModel
import torch
import torch.optim as optim
from tqdm import tqdm

device = 'cuda:3'
n_epochs = 15

train_data = DatasetH5('/scratch/arsh/mri_h5/scans/mri_data.hdf5', '/scratch/arsh/mri_h5/stimuli/stimuli_data.hdf5', 1, 'D', list(range(1, 14)), device=device)
train_generator = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = DatasetH5('/scratch/arsh/mri_h5/scans/mri_data.hdf5', '/scratch/arsh/mri_h5/stimuli/stimuli_data.hdf5', 1, 'D', [14, 15], device=device)
test_generator = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

mri_model = BrainModel()
mri_model.to(device)
print("Transferred model to cuda")


loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(mri_model.parameters(), lr=0.00003)

train_loss = []
test_loss = []

for epoch in range(n_epochs):
    train_loss.append([])
    for X_batch, y_batch in (pbar := tqdm(train_generator)):
        y_pred = mri_model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        pbar.set_description(f"Loss {loss}")
        train_loss[-1].append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        for X_batch, y_batch in test_generator:
            y_pred = mri_model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            test_loss.append(loss)
        print(f"Epoch {epoch}, Loss {sum(train_loss[-1])/len(train_loss[-1])}, Test Loss {test_loss[-1]}")

torch.save(mri_model.state_dict(), 'basic_model10.pt')

