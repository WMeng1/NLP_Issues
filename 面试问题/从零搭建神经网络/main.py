import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as dataLoader
from torchvision import datasets, transforms

epochs = 10
categroy = 10
lr = 1e-5
hidden_size = 200

class Net(nn.Module):
    def __init__(self, hidden_size, category) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, category)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

net = Net(hidden_size, categroy)
print(net)

optimizer = optim.Adam(net.parameters(), lr=lr)

criterion = nn.NLLLoss()


if __name__ == '__main__':
    train_loader = dataLoader.DataLoader(
        datasets.mnist('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3801,))
                       ])),
        batch_size=128, shuffle=True
    )

    test_loader = dataLoader.DataLoader(
        datasets.mnist('../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3801,))
                       ])),
        batch_size=128, shuffle=True
    )

    for epoch in range(epochs):
        for batch_idx, (train_data, target) in enumerate(train_loader):
            train_data = train_data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(train_data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}]'.format(
                    epoch, batch_idx * len(train_data), len(train_loader.dataset),
                    100.* batch_idx/len(train_loader), loss.data))

    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()