import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # first convolutional layer
        # input: (1, 28, 28) output: (32, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,          # number of channels in input
                out_channels=32,        # number of channels in output(filters number)
                kernel_size=5,          # the size of filter
                stride=1,               # the size of step
                padding=2,              # the size of area to fill zero around the output
            ),
            nn.ReLU(),                  # activation
        )
        # first max pooling layer
        # window size [2, 2]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # second convolutional layer
        # input: (32, 14, 14) output: (64, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # second max pooling layer
        # window size [2, 2]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # fully connected network
        # input: (64, 7, 7) output: 10 classes
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=300),
            nn.ReLU(),
            nn.Linear(300, 10),
            nn.LogSoftmax(dim=1)
        )

    def __call__(self, *args, **kwargs):
        return super(CNN, self).__call__(*args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        output = self.fc(x.view(-1, 64*7*7))
        return output

model = CNN()
print(model)

#########################################################################
#                           Parameters Settings                         #
#########################################################################
learning_rate = 0.001
batch_size = 50
epochs = 1
log_interval = 120
cuda_enable = True
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
    for batch_idx, (data, target) in enumerate(train_loader):

        # if cuda is enable, change the compute instances to cuda objects
        if cuda_enable:
            data, target = data.cuda(), target.cuda()

        # wraps tensors and can record the operation applied to it
        data, target = Variable(data), Variable(target)

        # define the optimizer and loss function
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss()

        # clear the gradient for each training step
        optimizer.zero_grad()

        # model output
        output = model(data)

        # compute loss and process back-propagation(compute gradient)
        loss = loss(output, target)
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


def test():

    # set the model to evaluation mode
    model.eval()

    # initialize loss and correct count
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if cuda_enable:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # sum up the loss
        test_loss += functional.cross_entropy(output, target, size_average=False).data[0]

        # get the index of the max log-probability
        predict = output.data.max(1, keepdim=True)[1]

        # counting correct predictions
        correct += predict.eq(target.data.view_as(predict)).cpu().sum()

    # get the average
    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )

###################################
# Running stage
for e in range(1, epochs + 1):
    train(e)

test()
