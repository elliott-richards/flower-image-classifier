import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
from workspace_utils import active_session

# Pre-load available architectures
alexnet = models.alexnet(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
# Create dictionary to convert input argument to model
model_dict = {'alexnet': alexnet, 'resnet18': resnet18, 'vgg16': vgg16}

# Create input arguments
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str, help = 'Directory containing the files to be trined on')
parser.add_argument('--save_dir', type = str, default='', help = 'Directory to save checkpoint in')
parser.add_argument('--arch', type = str, default='vgg16', 
                    help = 'CNN model architecture. Available models: alexnet, resnet18, vgg16')
parser.add_argument('--learning_rate', default=0.001, help = 'Gradient descent learn rate')
parser.add_argument('--hidden_units', default=4096, help = 'The  umber of hidden units in the classifier layer')
parser.add_argument('--epochs', default=5, help = 'The number of epochs to train over')
parser.add_argument('--gpu', type=bool, default=False, help = 'Enables the use of a GPU in running the model')

args = parser.parse_args()

# Create paths from input argument
train_path = args.data_dir + '/train'
valid_path = args.data_dir + '/valid'

# Compose transforms
train_transform = transforms.Compose([transforms.RandomRotation(15),
                                      transforms.RandomResizedCrop(224), 
                                      transforms.RandomHorizontalFlip(p=0.2), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(255), 
                                         transforms.CenterCrop(224), 
                                         transforms.ToTensor(), 
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Create data sets
train_set = datasets.ImageFolder(train_path, transform=train_transform)
valid_set = datasets.ImageFolder(valid_path, transform=valid_transform)
# Create data loaders
trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=32)

# Check if GPU is desired
if args.gpu:
    # Check if GPU is available
    if torch.cuda.is_available():
        device = 'cuda'
    # Notify if GPU is not available and use CPU instead
    else:
        print('GPU unavailable, using CPU instead')
        device = 'cpu'

# Initialise model
model = model_dict[args.arch]

for parameter in model.parameters():
    parameter.requires_grad = False

# Create classifier and add to model
classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, args.hidden_units), 
                           nn.ReLU(),
                           nn.Dropout(0.5), 
                           nn.Linear(args.hidden_units, 102), 
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

optimiser = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

criterion = nn.NLLLoss()

model.to(device)

# Train model
with active_session():
    for e in range(args.epochs):
        
        running_loss = 0

        for image, label in trainloader:
            
            image, label = image.to(device), label.to(device)
            
            optimiser.zero_grad()
            
            log_prob = model(image)
            loss = criterion(log_prob, label)
            loss.backward()
            
            running_loss += loss.item()
            
            optimiser.step()
            
        else:
            
            valid_loss = 0
            accuracy = 0
            # Put model in evaluate mode
            model.eval()
            
            with torch.no_grad():
                for image, label in validloader:
                    image, label = image.to(device), label.to(device)

                    log_prob = model(image)
                    valid_loss += criterion(log_prob, label).item()
                    prob = torch.exp(log_prob)
                    top_prob, top_class = prob.topk(1, dim=1)
                    match = top_class == label.view(*top_class.shape)
                    accuracy += torch.mean(match.type(torch.FloatTensor)).item()
            # Return model to trining mode
            model.train()
            
        print('Epoch: {:>2s} of {} | '.format(str(e+1), args.epochs), 
              'Training Loss: {:.3f} | '.format(running_loss / len(trainloader)), 
              'Validation Loss: {:.3f} | '.format(valid_loss / len(validloader)), 
              'Accuracy: {:.2f} %'.format(accuracy / len(validloader) * 100))
        
        
print('Training Complete')

# Save model
model.class_to_idx = train_set.class_to_idx
checkpoint = {'state_dict': model.state_dict(),
              'optimiser_state_dict': optimiser.state_dict(), 
              'classifier': model.classifier, 
              'epochs': args.epochs, 
              'hidden_units': args.hidden_units,
              'architecture': args.arch,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, args.save_dir + 'checkpoint.pth')
print('Checkpoint Saved')
              