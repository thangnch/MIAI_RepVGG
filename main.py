
import numpy
import time
import torch.utils.data.distributed
import torchvision.transforms as transforms

from repvgg import get_RepVGG_func_by_name
import ast
from utils import load_checkpoint
from PIL import Image
from torch.autograd import Variable


def image_loader(image_name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    image = Image.open(image_name)
    image = trans(image).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image

def load_imagenet_class_labels():
    file = open("imagenet1000_clsidx_to_labels.txt", "r")
    contents = file. read()
    image_labels = ast.literal_eval(contents)
    file.close()
    return image_labels

# Load labels image net
image_labels = load_imagenet_class_labels()

repvgg_build_func = get_RepVGG_func_by_name('RepVGG-A0')
model = repvgg_build_func(deploy=True)
load_checkpoint(model, "RepVGG-A0-deploy.pth")
model.eval()


image = image_loader("test_images\\n01806143_peacock.jpg")

with torch.no_grad():
    # compute output
    output = model(image)
    class_id = numpy.argmax(output.numpy())

    print("Class = ", image_labels[class_id])


