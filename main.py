from DataLoaderTorch import Dataset
from model import BrainModel
import torch
import torch.optim as optim
from tqdm import tqdm

device = 'cuda:3'

mri_data = Dataset('/scratch/arsh/mri_torch/scans', '/scratch/arsh/mri_torch/Encoded_Images_Kandinsky', 2, 'C', list(range(1, 14)), device=device)
test_mri_data = Dataset('/scratch/arsh/mri_torch/scans', '/scratch/arsh/mri_torch/Encoded_Images_Kandinsky', 2, 'C', [14, 15], device=device)

mri_model = BrainModel()
mri_model.to(device)
print("Transferred model to cuda")


loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(mri_model.parameters(), lr=0.00003)


# batch = next(mri_data)
# print(batch[0].shape)
# print(batch[0].max())
#
# print(batch[1].shape)
#
# output = mri_model.forward(batch[0])
# print(output.shape)

# loss = loss_fn(output, batch[1])

n_epochs = 15  # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = 50

# collect statistics
train_loss = []
test_loss = []

for epoch in range(n_epochs):
    train_loss.append([])
    for i in (pbar := tqdm(range(batches_per_epoch))):
        Xbatch, ybatch = next(mri_data)
        y_pred = mri_model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        pbar.set_description(f"Loss {loss}")
        # store metrics
        train_loss[-1].append(float(loss))
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # print progress
    with torch.no_grad():
        Xbatch, ybatch = next(test_mri_data)
        y_pred = mri_model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        test_loss.append(loss)
        print(f"Epoch {epoch}, Loss {sum(train_loss[-1])/len(train_loss[-1])}, Test Loss {loss}")

torch.save(mri_model.state_dict(), 'basic_model5.pt')
