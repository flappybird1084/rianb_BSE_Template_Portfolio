
<h1>304 | Facial Recognition System </h1>
<h3>By Rian Butala</h3>

[Github Repo](https://github.com/flappybird1084/rianb_BSE_Template_Portfolio/tree/gh-pages)<br>
[Code Repo](https://github.com/flappybird1084/bse_face_recognition)

<img src = "IMG_5561.jpeg" width = "250" height = "300">

<h1>Main Project</h1>
In this project, I will design a facial recognition system. 
Some major libraries were required for this process: facenet-pytorch, and pytorch. I also needed to reconfigure the python environments on my laptop with pyenv, as python had been installed on homebrew which blocked some libaries from being installed.
The main plan is as such: Take an image, isolate the most prominent face in it, and run image recognition <b>only on the face.</b> 

<!--Replace this text with a brief description (2-3 sentences) of your project. This description should draw the reader in and make them interested in what you've built. You can include what the biggest challenges, takeaways, and triumphs from completing the project were. As you complete your portfolio, remember your audience is less familiar than you are with all that your project entails! 


<iframe width="560" height="315" src="https://www.youtube.com/embed/X_XL_MmhUXI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

-->


| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Rian B | Stratford Preparatory | Computer Science | Incoming Junior

<img src = "Rian_B.JPG" width = "250" height = "300">

<!--
**Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.**

![Headstone Image](logo.svg)

-->
<h1>Milestone 1</h1>



<iframe width="560" height="315" src="https://www.youtube.com/embed/Yb5uKxvU8p0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<h2>Description</h2>
Getting to this stage was quite hard. A number of challenges caused me to be completely lost on many of the operations done up till this point. First, when installing libraries like pytorch, I had to completely reconfigure my python environment with pyenv. I also couldn't figure out how to load images into the transfer learning model on my own, so I downloaded pytorch's example notebook and modified it for my use case. However, the biggest issue was with the Apple MPS, or Metal Performance Shaders. The MPS is responisible for graphical compute and, in our case, accelerated performance with machine learning models (I only learned later that it provided a noticeable boost when running models, not training them). Documentation online for this was quite sparse, and I had to download the nightly build of pytorch, which was probably unnecessary, to train on the MPS, in addition to a few tweaks to number formatting for easier compute. Funnily enough, the MPS uses a proprietary type of float type called MPSFloatType, which is not compatible with cpu-bound models using torch.FloatTensor. This meant that the model I trained on my laptop couldn't be used on the raspberry pi, so I had to train another model on the CPU. It wasn't that bad, though, as I later ended up training many more models. 
<br><be>


<h2>Accomplishments</h2>
Some things that I accomplished in the process of reaching this milestone were:
<br>- Took majority of training and validation pictures
<br>- Learned how to use pytorch to transfer learn
<br>- Learned how to use the Pillow Imaging Libary
<br>- Learned how to use facenet-pytorch's MTCNN model to identify and track faces across different frames of photo and video
<br>- Cropped training and validation images to only the faces, eliminating unnecessary background clutter
<br>- Trained primary model on top of resnet18 (future ones to be trained on InceptionResnetv1)
<br>- Set up python3 virtual environments on a linux machine

<h2>Code</h2>

This project was very code-intensive. First, I had to take pictures using the OpenCV python library and save them to a specified path on my computer.
```
## Original source from here:
## https://realpython.com/face-detection-in-python-using-a-webcam/

import cv2
import time

video_capture = cv2.VideoCapture(0)
folder = "./img/train/notrian/"

counter = 0
while True:
    #print("loop started")
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        print("Exiting...")
        break
    if key  == ord('s'):
        timestr = time.strftime("%Y-%m-%d %H:%M:%S") 
        print("Saving image: "+timestr +"c"+str(counter)+ ".jpeg")
        filename = folder + timestr +"c"+str(counter)+ ".jpeg"
        cv2.imwrite(filename, frame)
        counter=counter+1
    if key == 13: ## enter key
        timestr = time.strftime("%Y-%m-%d %H:%M:%S")
        print("Saving image: "+timestr + ".jpeg")
        filename = folder + timestr + ".jpeg"
        cv2.imwrite(filename, frame)

video_capture.release()
cv2.destroyAllWindows()
```

Then, I needed to isolate all the faces from the pictures and save them separately, as the rest of the image was unnecessary. I ran this program in the form of a python notebook to monitor the code and debug faster, but for convenience have listed it all in a big chunk here.
```
from facenet_pytorch import MTCNN
import torch
import numpy as np
#import mmcv
import cv2
from PIL import Image, ImageDraw
from IPython import display
import os,glob

device = torch.device('cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

path = "/Users/rianbutala/Rian's projects/Coding/Facial Recognition/img/train/rian"
save_path = "/Users/rianbutala/Rian's projects/Coding/Facial Recognition/cropped/train/rian"

os.chdir(path)

images = []

for file in glob.glob("*.jpeg"):
    with Image.open(file) as img:
        #img.show()
        images.append(file)
print(type(images[0]))


imgs =[]
#print('\rTracking frame: {}'.format(i + 1), end='')

# Detect faces
for count,img in enumerate(images):
    os.chdir(path)
    print(img)
    img = Image.open(img)
    boxes, _ = mtcnn.detect(img)

    # Draw faces
    frame_draw = img.copy()
    draw = ImageDraw.Draw(frame_draw)
    try:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            print(box)

        cropped_img = img.copy()
        cropped_img = cropped_img.crop(boxes[0])
        #d = display.display(cropped_img, display_id=True)
        os.chdir(save_path)
        cropped_img.save("img_"+str(count)+".jpeg")
        print("saved!!")


        # Add to frame list
        imgs.append(frame_draw.resize((640, 360), Image.BILINEAR))
        print('\nDone')
    except:
        print("caught!!")


#d = display.display(imgs[0], display_id=True)
for img in imgs:
    display.display(img,display_id=True)

```

After that, I trained the model by transfer learning on resnet-18 (also imported from notebook). I had a lot of trouble trying to use Apple's MPS backend to train the model, as Pytorch had nicer compatibility with CPU and Nvidia CUDA training, but managed to figure it out in the end. However, MPS-trained models cannot be run on the CPU, so I had to redo my training all on the CPU.

```
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# License: BSD

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'cropped'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=14) #changed from batch size 4 and num workers 4
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names[0])

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)


#show a few images
imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
    
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.float() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(weights='IMAGENET1K_V1')
#model_ft = facenet_pytorch.InceptionResnetV1(pretrained='vggface2', device=device, classify= True, num_classes=1)


num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.

model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

visualize_model(model_ft)

torch.save(model_ft, "model_ft.pt")
```

At the end of that file, we save the model as 'model_ft.pt' for later use, and deployment on the Raspberry Pi, as I obviously can't train the model on it. 
Here's a small piece of code that lets me test the trained model on the same set of images.

```
# License: BSD

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "/Users/rianbutala/Rian's projects/Coding/Facial Recognition/cropped"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=14) #changed from batch size 4 and num workers 4
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


model_ft = torch.load("model_ft.pt")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('mps:0')

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(model_ft)
```
Now, let's start using the model with the camera stream. Because the previous block of code ran the recognizer on a precropped image, we'll need to develop a new program to take an image file, crop it, and run predictions. 
```
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory
from facenet_pytorch import MTCNN
import numpy

video_capture = cv2.VideoCapture(0)

model_ft = torch.load("model_ft_3.pt")

device = torch.device('cpu')

mtcnn = MTCNN(keep_all=True, device=device)

image_path = ""

def pre_image(model, image_path):
    #img = Image.open(image_path)
    cv2img =cv2.imread(image_path)

    #_,cv2img = video_capture.read()
    boxes, _ = mtcnn.detect(cv2img)

    color_converted = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB) 

    pilcv2img = Image.fromarray(color_converted).copy()
   
    frame_draw = pilcv2img.copy()
    draw = ImageDraw.Draw(frame_draw)


    try:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            print(box)
        #frame_draw.show()
        #cv2.imshow("frame2", cv2img)
        #key = cv2.waitKey(1) & 0xff

        cropped_img = pilcv2img.copy()
        cropped_img = cropped_img.crop(boxes[0])
    except:
        print("caught!!")
        cropped_img = pilcv2img.copy()
        

    
    img = frame_draw.copy()
    open_cv_image = numpy.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    img2 = frame_draw.copy()
    try:
        img2 = img2.crop(boxes[0])
    except:
        pass
    open_cv_image2 = numpy.array(img2)
    open_cv_image2 = open_cv_image2[:, :, ::-1].copy()



    cv2.imshow("rect-frame", open_cv_image)
    key = cv2.waitKey(1) & 0xff


    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(), 
    transforms.Resize((224,224)),transforms.Normalize(mean, std)])
    # get normalized image

    try:
        img_normalized = transform_norm(img.crop(boxes[0])).float()
        img_normalized = img_normalized.unsqueeze_(0)
        # input = Variable(image_tensor)
        img_normalized = img_normalized.to(device)
        # print(img_normalized.shape)
       
    except:
        img_normalized = transform_norm(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        # input = Variable(image_tensor)
        img_normalized = img_normalized.to(device)
        print("didn't crop")
        # print(img_normalized.shape)

    with torch.no_grad():
        model.eval()  
        output =model(img_normalized)
        print(output)
        index = output.data.cpu().numpy().argmax()
        print(index)
        #classes = train_ds.classes
        #class_name = classes[index]
        #return class_name

_,img = video_capture.read()
#cv2.imshow("Frame",img)

pre_image(model_ft, "/Users/rianbutala/Rian's projects/Coding/Facial Recognition/bse_face_recognition/tester_image.jpeg")
    #time.sleep(0.1)
#time.sleep(10)
cv2.waitKey(0)

cv2.destroyAllWindows()
```
And now, with a few modifications, we can run the model on a video stream:
```
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory
from facenet_pytorch import MTCNN
import numpy

video_capture = cv2.VideoCapture(0)

model_ft = torch.load("model_ft_4.pt")

device = torch.device('cpu')

mtcnn = MTCNN(keep_all=True, device=device)

def pre_image(model):
    #img = Image.open(image_path)
    _,cv2img = video_capture.read()
    boxes, _ = mtcnn.detect(cv2img)

    color_converted = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB) 

    pilcv2img = Image.fromarray(color_converted).copy()
   
    frame_draw = pilcv2img.copy()
    draw = ImageDraw.Draw(frame_draw)


    try:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            print(box)
        #frame_draw.show()
        #cv2.imshow("frame2", cv2img)
        #key = cv2.waitKey(1) & 0xff

        cropped_img = pilcv2img.copy()
        cropped_img = cropped_img.crop(boxes[0])
    except:
        print("caught!!")
        cropped_img = pilcv2img.copy()
        

    
    img = frame_draw.copy()
    open_cv_image = numpy.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    img2 = frame_draw.copy()
    try:
        img2 = img2.crop(boxes[0])
    except:
        pass
    open_cv_image2 = numpy.array(img2)
    open_cv_image2 = open_cv_image2[:, :, ::-1].copy()



    cv2.imshow("rect-frame", open_cv_image)
    key = cv2.waitKey(1) & 0xff


    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(), 
    transforms.Resize((224,224)),transforms.Normalize(mean, std)])
    # get normalized image

    try:
        img_normalized = transform_norm(img.crop(boxes[0])).float()
        img_normalized = img_normalized.unsqueeze_(0)
        # input = Variable(image_tensor)
        img_normalized = img_normalized.to(device)
        # print(img_normalized.shape)
       
    except:
        img_normalized = transform_norm(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        # input = Variable(image_tensor)
        img_normalized = img_normalized.to(device)
        print("didn't crop")
        # print(img_normalized.shape)

    with torch.no_grad():
        model.eval()  
        output =model(img_normalized)
        print(output)
        index = output.data.cpu().numpy().argmax()
        print(index)
        #classes = train_ds.classes
        #class_name = classes[index]
        #return class_name

_,img = video_capture.read()
#cv2.imshow("Frame",img)

while(True):
    pre_image(model_ft)
    #time.sleep(0.1)
#time.sleep(10)
cv2.waitKey(0)

cv2.destroyAllWindows()
```

<h1>Starter Project</h1>

<iframe width="560" height="315" src="https://www.youtube.com/embed/X_XL_MmhUXI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

My starter project was Weevil Eyes. In this project, I soldered a couple individual components onto a circuit board. 
The end product is a tiny device with LEDs and a photoresistor that lights up when it senses a lack of light around it. 
This works with a battery providing power to LEDs and a transistor blocking the flow of power until the photoresistor resists nothing. This typically occurs when there is no light falling on it. 


<h2> Bill of Materials</h2>


| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| CanaKit Raspberry Pi 4 Starter Kit | Image processing | $119.99 | <a href="https://www.canakit.com/raspberry-pi-4-starter-kit.html"> Link </a> |
| Haiway 10.1 Inch Monitor| Initial setup of Pi and minor adjustments | $84.96 | <a href="https://www.amazon.com/Haiway-Security-Surveillance-Controller-Resolution/dp/B07WCQ627G?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&smid=A2ECNUTN5RLACN&th=1"> Link </a> |
| Logitech K120 Keyboard | Basic input | $12.34 | <a href="https://www.amazon.com/Logitech-920-002478-K120-USB-Keyboard/dp/B003ELVLKU/ref=sr_1_2?crid=3QE83NKLCZ1UB&dib=eyJ2IjoiMSJ9.hGINAjjbAmcnMmhSu62W7ybtCHaT8ifr068BE_xt70sqaJKSERXvtm9l4hcYzEzb1Nadmebc8KfMnVBUMOHJ_fo_kXmFEZ2vVP70KkO0JfP_imqKzqrFmr2PcwG1egHFtqYNIuwuBlGPaihgt6WzWLyBDvc2R7EMOPgLOsKY1VU-SgHs18jkv59qxYLWyeCnfLo88_cstvYpQygQcHqg05iKghON5vYXtHjiUaHM45dKs2eBoMiAmUKu09tcs6j93HJBOjJSF850VYd05UpW1PSgwPFabdDnlwHm7-xTAHk.9L3Pp4zftTMPPQaXuohixdM7KOBZGOdJMPSt1_RXcTk&dib_tag=se&keywords=k120&qid=1718309899&s=electronics&sprefix=k120%2Celectronics%2C142&sr=1-2&th=1"> Link </a> |
| Logitech B100 Mouse | Basic input | $7.99 | <a href="https://www.amazon.com/Logitech-B100-Corded-Mouse-Computers/dp/B003L62T7W/ref=sr_1_3?crid=2GLUL6WJZ0GKO&dib=eyJ2IjoiMSJ9.OKAfwMtMmgjzpEXrJp10_w8xKaMtq7qsCFw-slfV25FJ6ELYelI8G81zHARc8xMbnTCq0tL_OChdFmyNgEhRPUoxERchBVR8gjhwMqhTFISEKzIPDAg4q4-67bUtJ9QuR-JyYdy4QKrLb_eqwybdizcPq1iZbiJ7LZNoIMVa6qXXi_bSBFNF3n90LwKkWHf0m7aNz-YVruwux6_LjHomLs7nRuOJZq9HAm_VWolRxoC5zXDEE4HmjvR3PZX3RyQ3xegWDG9tbdSWfNdPpzPbkSI2vTYmhCHUokWoZ-0Po-g.clnJylxu8WzF8OdB44ZsJ2Hct5aYZCMng0HDx-eXojM&dib_tag=se&keywords=mouse%2Blogitech%2Bwired&qid=1718309962&s=electronics&sprefix=mouse%2Blogitech%2Bwir%2Celectronics%2C137&sr=1-3&th=1"> Link </a> |


<!--  
# Final Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/F7M7imOVGug" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE



# Second Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/y3VAmNlER5Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your second milestone, explain what you've worked on since your previous milestone. You can highlight:
- Technical details of what you've accomplished and how they contribute to the final goal
- What has been surprising about the project so far
- Previous challenges you faced that you overcame
- What needs to be completed before your final milestone 

# First Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/CaCazFBhYKs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your first milestone, describe what your project is and how you plan to build it. You can include:
- An explanation about the different components of your project and how they will all integrate together
- Technical progress you've made so far
- Challenges you're facing and solving in your future milestones
- What your plan is to complete your project

# Schematics 
Here's where you'll put images of your schematics. [Tinkercad](https://www.tinkercad.com/blog/official-guide-to-tinkercad-circuits) and [Fritzing](https://fritzing.org/learning/) are both great resoruces to create professional schematic diagrams, though BSE recommends Tinkercad becuase it can be done easily and for free in the browser. 

# Code
Here's where you'll put your code. The syntax below places it into a block of code. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize it to your project needs. 

```c++
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("Hello World!");
}

void loop() {
  // put your main code here, to run repeatedly:

}
```

# Bill of Materials
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |

# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.

-->
