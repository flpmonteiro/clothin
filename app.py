from typing import List, Tuple, Union

import streamlit as st

from PIL import Image
import torch
from torch import nn
from torchvision.transforms import v2
from torchvision import models


def predict(
    model: nn.Module,
    image: Image.Image,
    image_size: Tuple[int, int] = (224, 224),
    transform=None,
    device: Union[None, str, torch.device] = "cpu",
) -> int:
    if transform is not None:
        image_transform = transform
    else:
        image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    model.to(device)
    model.eval()

    with torch.inference_mode():
        transformed_image = image_transform(image).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    return torch.argmax(target_image_pred, dim=1).item()


model = models.efficientnet_b0()
model.classifier = nn.Sequential(
    nn.Dropout(), nn.Linear(in_features=1280, out_features=10)
)

model.load_state_dict(torch.load("model0.pth", map_location=torch.device("cpu")))


st.title("Image classification with Streamlit")

col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("Upload an image", type="jpg")

class_names = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption="Uploaded image", use_column_width=True)

    predictions = predict(
        model=model,
        image=image,
    )
    col2.header("Product")

    col2.text_input(
        label="Title",
        placeholder="eg: 'summer dress' or 'running shirt'",
        max_chars=None,
    )

    col2.text_input(
        label="brand", placeholder="What's the product's brand?", max_chars=None
    )
    col2.checkbox(label="No brand", value=False)

    col2.text_area(
        label="description",
        placeholder="eg: 'Vintage floral dress, size S. Good condition with minor wear. Great for retro-themed parties!'",
    )

    col2.checkbox(label="Used?", value=True)

    col2.selectbox(
        label="category",
        options=class_names,
        index=predictions,
        label_visibility="visible",
    )
    col2.selectbox(
        label="sub-category",
        options=class_names,
        index=predictions,
        label_visibility="visible",
    )
    col2.selectbox(
        label="characteristics",
        options=class_names,
        index=predictions,
        label_visibility="visible",
    )
    col2.selectbox(
        label="department",
        options=class_names,
        index=predictions,
        label_visibility="visible",
    )
