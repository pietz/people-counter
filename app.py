import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs

from u2net import U2NETP, U2NET

@st.cache
def load_model(path):
    model = U2NETP()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict_patches(model, x, split=2):
    c, h, w = x.size()
    p = torch.zeros_like(x)
    for row in range(0, h, h//split):
        for col in range(0, w, w//split):
            print(row, col)
            p[:,row:row+h//split,col:col+w//split] = model(x[:,row:row+h//split,col:col+w//split].unsqueeze(0))[0].squeeze(0)
    return p


transform = tfs.ToTensor()
model = load_model('U2NETP_STPB_8.9MAE.pt')


st.title('Let me count those people for you')
file = st.sidebar.file_uploader('Choose a photo of people')

if file is not None:
    pil_img = Image.open(file).convert('RGB')
    st.sidebar.image(pil_img, use_column_width=True)
    width = st.sidebar.number_input('Image Width', 240, 1024, 720)
    if pil_img.size[0] > width:
        w, h = pil_img.size
        pil_img = pil_img.resize((width, int(h/w*width)))
    x = transform(pil_img)
    c, h, w = x.size()
    with torch.no_grad():
        p = model(x.unsqueeze(0))[0].squeeze(0).squeeze(0)
    #p = predict_patches(model, x, 2)
    count = int(round(int(p.sum())/100)) # model is trained with 100x higher values
    p = (p-p.min()) / (p.max()-p.min()) * 255
    p = p.detach().numpy().astype(np.uint8)
    p_img = Image.fromarray(p)
    st.image(p_img)
    st.header("I'm counting " + str(count) + " people in this photo.")