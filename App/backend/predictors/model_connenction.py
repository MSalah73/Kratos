import cv2
import torch
import torchvision
import os

dir = os.getcwd()
color = []
classes_path = dir+"/deep-fashion/colors/allcolor.txt"
with open(classes_path) as openfileobject:
            for line in openfileobject:
                color.append(line)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1_bn = torch.nn.BatchNorm2d(48)
        self.conv2_bn = torch.nn.BatchNorm2d(64)
        self.conv5_bn = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(3, 48, kernel_size=11)
        self.conv2 = torch.nn.Conv2d(48, 64, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(64,192, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(192, 96, kernel_size=3)
        self.conv5 = torch.nn.Conv2d(96, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64*9*9, 4096)
        self.fc2 = torch.nn.Linear(4096, 961)
        # end
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1_bn(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2_bn(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=1)
        #x1, x2 = torch.split(x, 96)
        #print(x1.shape)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv5_bn(x)
        x = torch.nn.functional.relu(x)
        #x = torch.flatten(x)
        x = x.view(-1, 5184)
        x = self.fc1(x)
        x = torch.nn.functional.dropout(x, p=0.35, training=self.training)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def prepare(file):
    IMG_SIZE = 112
    img_array = cv2.imread(file, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = torchvision.transforms.functional.to_tensor(new_array)
    new_array.unsqueeze_(0)
    return new_array

def predict(filename):
    output = moduleNetwork(prepare(filename))
    output = output.data.max(dim=1, keepdim=False)[1]
    prediction = color[output]
    temp = {}
    temp["name"] = "Yikun"
    temp["type"] = "color"
    temp["prediction"] = prediction
    prediction = temp
    return prediction


moduleNetwork = Network()
moduleNetwork.load_state_dict(torch.load(dir+'/models/test.pth'))
moduleNetwork.eval()
