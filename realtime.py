import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms

classes=('with_mask','without_mask')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(

            nn.Conv2d(3, 6, 5),  # (N, 3, 224, 224) -> (N,  6, 220, 220)
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 220, 220) -> (N,  6, 110, 110)
            nn.Conv2d(6, 16, 5),  # (N, 6, 110, 110) -> (N, 16, 106, 106)
            #nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)  # (N,16, 106, 106) -> (N, 16, 53, 53)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(44944, 120),  # (N, 400) -> (N, 120)
            #nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(120, 84),  # (N, 120) -> (N, 84)
            nn.Tanh(),
            nn.Linear(84, 2)  # (N, 84)  -> (N, 2)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # -1, in_features x.size(0), -1
        x = self.fc_model(x)
        return x



#net.load_state_dict(torch.load(PATH))
model=torch.load("models/mask_model_lenet_20epoch.pth")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # pil_image = Image.open(image)
    pil_image = image

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = image_transforms(pil_image)
    return img




def classify_face(image):
    # device = torch.device("cpu")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # im_pil = image.fromarray(image)
    # image = np.asarray(im)
    im = Image.fromarray(image)
    image = process_image(im)
    print('image_processed')
    img = image.unsqueeze_(0)
    img = image.float()

    # model.eval()
    # model.cpu()
    output = model(image)
    print(output ,'##############output###########')
    _, predicted = torch.max(output, 1)
    print(predicted.data[0] ,"predicted")


    classification1 = predicted.data[0]
    index = int(classification1)
    print(classes[index])
    return classes[index]


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2
#faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    label = classify_face(frame)
    if(label == 'with_mask'):
        print("No Beep")
    else:
        #sound.play()
        print("Beep")
    cv2.putText(frame,str(label),(100,height-20), font, 1,(0,255,0),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


