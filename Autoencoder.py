import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def __call__(self, *args, **kwargs):
        return super(Autoencoder, self).__call__(*args, **kwargs)

    def forward(self, x):
        return self.decoder(self.encoder(x.view(-1, 784)))

model = Autoencoder()
print(model)

#########################################################################
#                           Parameters Settings                         #
#########################################################################
learning_rate = 0.003
batch_size = 50
epochs = 5
log_interval = 120
cuda_enable = True
test_img = 15
#########################################################################

cuda_enable = torch.cuda.is_available() and cuda_enable
if cuda_enable:
    model.cuda()


train_data = datasets.MNIST(
    root='./mnist/',                    # the path where the data to store
    train=True,
    transform=transforms.ToTensor(),    # converts a PIL.Image or numpy.ndarray instance to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,                      # to download the dataset from Internet
)
test_data = datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


def train(epoch):

    # set the model to training mode
    model.train()

    # training step
    for batch_idx, (data, _) in enumerate(train_loader):

        # if cuda is enable, change the compute instances to cuda objects
        if cuda_enable:
            data = data.cuda()

        # wraps tensors and can record the operation applied to it
        data = Variable(data)

        # define the optimizer and loss function
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        loss = nn.MSELoss()

        # clear the gradient for each training step
        optimizer.zero_grad()

        # model output
        output = model(data)

        # compute loss and process back-propagation(compute gradient)
        loss = loss(output, data)
        loss.backward()

        # apply the gradients
        optimizer.step()

        # print the progress
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0]
                )
            )


def test(epoch):

    # set the model to evaluation mode
    model.eval()

    # initialize the test loss
    test_loss = 0

    for i, (data, _) in enumerate(test_loader):
        if cuda_enable:
            data = data.cuda()

        data = Variable(data, volatile=True)
        output = model(data)

        # sum up the loss
        test_loss += functional.mse_loss(output, data).data[0]

        # when constructing autoencoder
        # the evaluation way is to generate the reconstructed images to compare with origin ones
        if i == 0:
            # use min(batch_size, test_img) as the image number in single output
            n = min(data.size(0), test_img)
            comparison = torch.cat([data[:n], output.view(batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


###################################
# Running stage
for e in range(1, epochs+1):
    train(e)
    test(e)
