import streamlit as st
#import numpy as np
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

width = 1024
transform = tfs.ToTensor()
model = load_model('U2NETP_STPB_11.7MAE.pt')


st.title('Let me count those people for you')
file = st.file_uploader('Choose a photo of people')

if file is not None:
    pil_img = Image.open(file).convert('RGB')
    if pil_img.size[0] > width:
        w, h = pil_img.size
        pil_img = pil_img.resize((width, int(h/w*width)))
    st.image(pil_img, use_column_width=True)
    x = transform(pil_img)
    p = model(x.unsqueeze(0))[0].squeeze(0)
    count = int(round(int(p.sum())/100)) # model is trained with 100x higher values
    st.header("I'm counting " + str(count) + " people in this photo.")