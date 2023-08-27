import gradio as gr
import torch
from PIL import Image
from Dataset.testalbumentation import TestAlbumentation
from Model.Lit_cifar_module import LitCifar
from utils import *

net = LitCifar().cpu()
net.load_state_dict(torch.load('final_dict.pth', map_location=torch.device('cpu')))
net.eval() 

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
global_classes = 5

def inference(input_image, transparency, target_layer, num_top_classes1, gradcam_image_display = False):
 im = input_image
 test_transform = TestAlbumentation()
 im1 = test_transform(im)
 im1 = im1.unsqueeze(0).cpu()
 out0 = net(im1)
 out = out0.detach().numpy()
 confidences = {classes[i] : float(out[0][i]) for i in range(10)}
 val = torch.argmax(out0).detach().numpy().tolist()
 targ = [val]
 input_image_np,visualization=gradcame(net, 0, targ, im1, target_layer, transparency)
 return confidences, visualization
 
interface = gr.Interface(inference, 
                         inputs = [gr.Image(shape=(32,32), type="pil", label = "Input image"), 
                         gr.Slider(0,1, value = 0.5, label="opacity"), 
                         gr.Slider(-2,-1, value = -2, step = 1, label="gradcam layer"),
                         gr.Slider(0,9, value = 0, step = 1, label="no. of top classes to display"), 
                         gr.Checkbox(default=False, label="Show Gradcam Image")],
                         outputs = [gr.Label(num_top_classes=global_classes),
                         gr.Image(shape=(32,32), label = "Output")],
                         title = "Gradcam output of network trained on cifar10",
                         examples = [["cat.jpg", 0.5, -1], ["dog.jpg",0.5,-1]],
                         )


# Launch the Gradio interface
interface.launch()