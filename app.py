import torch
import snntorch
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_per_process_memory_fraction(0.85, device=0)



import numpy as np

import tonic
from tonic import datasets, transforms
from torch.utils.data import DataLoader
from collections import namedtuple

State = namedtuple("State", "obs labels")




class _SHD2Raster():
    """ 
    Tool for rastering SHD samples into frames. Packs bits along the temporal axis for memory efficiency. This means
        that the used will have to apply jnp.unpackbits(events, axis=<time axis>) prior to feeding the data to the network.
    """

    def __init__(self, encoding_dim, sample_T = 100):
        self.encoding_dim = encoding_dim
        self.sample_T = sample_T
        
    def __call__(self, events):
        # tensor has dimensions (time_steps, encoding_dim)
        tensor = np.zeros((events["t"].max()+1, self.encoding_dim), dtype=int)
        np.add.at(tensor, (events["t"], events["x"]), 1)
        #return tensor[:self.sample_T,:]
        tensor = tensor[:self.sample_T,:]
        tensor = np.minimum(tensor, 1)
        #tensor = np.packbits(tensor, axis=0) pytorch does not have an unpack feature.
        return tensor


sample_T = 128
shd_timestep = 1e-6
shd_channels = 700
net_channels = 128
net_dt = 1/sample_T
batch_size = 256

obs_shape = tuple([net_channels,])
act_shape = tuple([20,])

transform = transforms.Compose([
    transforms.Downsample(
        time_factor=shd_timestep / net_dt,
        spatial_factor=net_channels / shd_channels
    ),
    _SHD2Raster(net_channels, sample_T=sample_T)
])

train_dataset = datasets.SHD("./data", train=True, transform=transform)
test_dataset = datasets.SHD("./data", train=False, transform=transform)

train_dl = iter(DataLoader(train_dataset, batch_size=len(train_dataset),
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
x_train, y_train = next(train_dl)
x_train, y_train = x_train.to(torch.uint8), y_train.to(torch.uint8)
x_train, y_train = x_train.to(device), y_train.to(device)

def shuffle(dataset):
    x, y = dataset

    cutoff = y.shape[0] % batch_size

    indices = torch.randperm(y.shape[0])[:-cutoff]
    obs, labels = x[indices], y[indices]


    obs = torch.reshape(obs, (-1, batch_size) + obs.shape[1:])
    labels = torch.reshape(labels, (-1, batch_size)) # should make batch size a global

    return State(obs=obs, labels=labels)


test_dl = iter(DataLoader(test_dataset, batch_size=len(test_dataset),
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
x_test, y_test = next(test_dl)
x_test, y_test = x_test.to(torch.uint8), y_test.to(torch.uint8)
x_test, y_test = x_test.to(device), y_test.to(device)
x_test, y_test = shuffle((x_test, y_test))


num_hidden = 64
# Define Network
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = torch.nn.Linear(128, num_hidden)
        self.lif1 = snntorch.Leaky(beta=torch.ones(num_hidden)*0.5, learn_beta=True)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)
        self.lif2 = snntorch.Leaky(beta=torch.ones(num_hidden)*0.5, learn_beta=True)
        self.fc3 = torch.nn.Linear(num_hidden, 20)
        self.lif3 = snntorch.Leaky(beta=torch.ones(20)*0.5, learn_beta=True, reset_mechanism="none")

    def forward(self, x):

        x = x.float() # [batch, time, channel]
        
        x = x.permute(1,0,2) # [time, batch, channel]
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        V = []

        # need to fix since data is not time leading axis...
        for i, step in enumerate(x):
            cur1 = self.fc1(step)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            V.append(mem3)

        
        return torch.stack(V, axis=0).permute(1,0,2)
        
# Load the network onto CUDA if available
net = Net().to(device)
#precompiled_net = Net().to(device)
#net = torch.compile(precompiled_net, fullgraph=True)


loss = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
acc = lambda predictions, targets : (torch.argmax(predictions, axis=-1) == targets).sum().item() / len(targets)



num_epochs = 50
loss_hist = []
counter = 0


# Outer training loop
for epoch in range(num_epochs):    
    print(epoch)
    train_batch = shuffle((x_train, y_train))
    train_data, targets = train_batch
    
    
    # Minibatch training loop
    for data, targets in zip(train_data, targets):

        # forward pass
        net.train()
        out_V = net(data)
        # initialize the loss & sum over time
        loss_val = loss(torch.sum(out_V, axis=-2), targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    # Store loss history for future plotting
    loss_hist.append(loss_val.item())


# Test set
with torch.no_grad():
    denominator = y_test[0]
    test_acc = 0
    batch_acc = []
    for test_data, test_targets in zip(x_test, y_test):
        net.eval()
        # Test set forward pass
        out_V = net(test_data)
        # Test set loss
        batch_acc.append( acc(torch.sum(out_V, axis=-2), test_targets) )
    
    test_acc = np.mean(batch_acc)


test_acc



num_epochs = 40
loss_hist = []
counter = 0


def _train(mod, data, targets):
    # forward pass

    out_V = mod(data)
    # initialize the loss & sum over time
    loss_val = loss(torch.sum(out_V, axis=-2), targets)
    # Gradient calculation + weight update
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    return loss_val

train = torch.compile(_train, mode="reduce-overhead")

net.train()

# Outer training loop
for epoch in range(num_epochs):    
    print(epoch)
    train_batch = shuffle((x_train, y_train))
    train_data, targets = train_batch
    
    
    # Minibatch training loop
    for data, target in zip(train_data, targets):
        data = data.to(torch.float16)
        target = target.to(torch.int64)
        # forward pass
        train(net, data, target)

    # Store loss history for future plotting


# Test set
with torch.no_grad():
    denominator = y_test[0]
    test_acc = 0
    batch_acc = []
    for test_data, test_targets in zip(x_test, y_test):
        net.eval()
        # Test set forward pass
        out_V = net(test_data)
        # Test set loss
        batch_acc.append( acc(torch.sum(out_V, axis=-2), test_targets) )
    
    test_acc = np.mean(batch_acc)
