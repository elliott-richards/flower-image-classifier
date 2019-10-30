import argparse
import torch
import json
import numpy as np
from torchvision import models
from PIL import Image

# Pre-load available architectures
alexnet = models.alexnet(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
# Create dictionary to convert input argument to model
model_dict = {'alexnet': alexnet, 'resnet18': resnet18, 'vgg16': vgg16}

# Create input parameters
parser = argparse.ArgumentParser()

parser.add_argument('path', type=str, help='Path to image')
parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Top K most likely cases')
parser.add_argument('--category_names', help='Path to index to category name mapper')
parser.add_argument('--gpu', type=bool, default=False, 
                    help='Enables the use of a GPU in running the model, \
                    should be set to either True or False (default False)')

args = parser.parse_args()

# Load checkpoint
checkpoint = torch.load(args.checkpoint)
# Load category names dictionary
with open(args.category_names, 'r') as file:
    class_to_name = json.load(file)

# Check if GPU is desired
if args.gpu:
    # Check if GPU is available
    if torch.cuda.is_available():
        device = 'cuda'
    # Notify if GPU is not available and use CPU instead
    else:
        print('GPU unavailable, using CPU instead')
        device = 'cpu'

# Load model from checkpoint
model = model_dict[checkpoint['architecture']]
model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

## Process image
# Open image
image = Image.open(args.path)
# Make shortest side 256 px
if image.width > image.height:
    image.thumbnail((np.inf, 256))
else:
    image.thumbnail((256, np.inf))
# Crop image    
left = int((image.width - 224) / 2)
right = left + 224
top = int((image.height - 224) / 2)
bottom = top + 224

image = image.crop((left, top, right, bottom))

# Convert to numpy array
np_image = np.array(image)
# Normalise
np_image = np_image / 255
np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
# Transpose for use in model
np_image = np_image.transpose((2, 0, 1))
# Create image tensor
image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
image_tensor.unsqueeze_(0)
image_tensor = image_tensor.to(device)

## Predict on image
# Obtain probabilities and classes
log_prob = model(image_tensor)
prob = torch.exp(log_prob)
top_prob, top_class = prob.topk(args.top_k, dim=1)
# Invert class to index dictionary
idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}

# Convert from indices to flower names 
top_prob_list = top_prob.cpu().detach().numpy().tolist()[0]
top_class_list = top_class.cpu().detach().numpy().tolist()[0]
top_class_class = [idx_to_class[i] for i in range(args.top_k)]
top_class_name = [class_to_name[top_class_class[i]] for i in range(args.top_k)]

# Print results
print('Results:')
for i in range(args.top_k):
    print('Flower: {:25s} Probability: {:.2f}'.format(top_class_name[i], top_prob_list[i] * 100))
