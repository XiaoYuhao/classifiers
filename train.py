import torch
import torchvision
import numpy as np 
import os, random
from xml.dom.minidom import Document, parse
from PIL import Image
from torchvision.transforms import functional as F
import time
from model import *
from dataset import CLASSES

def readXml(filepath):
    domTree = parse(filepath)
    rootNode = domTree.documentElement
    object_node = rootNode.getElementsByTagName("class")[0]
    object_cls = object_node.childNodes[0].data
    return object_cls

def nll_loss(log_softmax, labels):
    loss_fn = torch.nn.NLLLoss(reduction="mean")
    loss = loss_fn(log_softmax, labels)
    #print(loss)
    return loss

def train_mbgd():
    Net = MobileNetV3_Large_Classification(class_num=102)
    print(Net)
    #in_channels = Net.fc.in_features
    #Net.fc = nn.Linear(in_channels, 102)
    #Net = ResNet_Offical()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    Net.to(device)
    img_list = sorted(os.listdir("./dataset/images"))
    anno_list = sorted(os.listdir("./dataset/annotations"))

    optimizer = torch.optim.SGD(Net.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
    batch_size = 20
    num_epochs = 20

    for epoch in range(num_epochs):
        for idx in range(0,7000,batch_size):
            img_path  = [os.path.join("./dataset/images", img_list[i]) for i in range(idx,idx+batch_size)]
            anno_path = [os.path.join("./dataset/annotations", anno_list[i]) for i in range(idx,idx+batch_size)]
            img  = [Image.open(path).convert("RGB") for path in img_path]
            anno = [CLASSES.index(readXml(path)) for path in anno_path]
            labels = torch.autograd.Variable(torch.Tensor(anno)).to(device).long()
            #func = ToTensor()
            #img, labels = func(img, labels)
            #img = img.to(device)
            img_var = [F.to_tensor(image) for image in img]
            img = torch.stack(img_var, dim=0).to(device)
            x = Net.forward(img)
            loss = nll_loss(x, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[%d]loss: %f" %(idx, loss))
        print("epoch[%d] finished." %epoch)

    torch.save(Net.state_dict(), "./weights/mobileNetV3_Large_20_model.pth")

def evaluate():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print(device)
    #Net = MobileNet_Classification(class_num=102)
    Net = MobileNetV3_Large_Classification(class_num=102)
    Net.load_state_dict(torch.load("./weights/mobileNetV3_Large_20_model.pth"))
    Net.eval()
    Net.to(device)

    img_list = sorted(os.listdir("./dataset/images"))
    anno_list = sorted(os.listdir("./dataset/annotations"))

    num = 0
    acc_num = 0

    start_time = time.time()

    for idx in range(7000, len(img_list)):
        img_path = os.path.join("./dataset/images", img_list[idx])
        anno_path = os.path.join("./dataset/annotations", anno_list[idx])
        labels = CLASSES.index(readXml(anno_path))
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)
        img_var = torch.unsqueeze(img, dim=0).to(device)

        x = Net.forward(img_var)
        pred_cls = torch.argmax(x).item()
        print(idx,CLASSES[pred_cls])
        if labels == pred_cls:
            acc_num += 1
        num += 1
    
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost time: %f" %cost_time)
    print("pre image cost time: %f" %(cost_time/num))
    print(acc_num)
    print(num)
    print(acc_num/num)


if __name__ == '__main__':
    #train_mbgd()
    evaluate()
