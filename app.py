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
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    transform=None,
    device: Union[None, str, torch.device] = "cpu",
) -> str:
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

    return class_names[torch.argmax(target_image_pred, dim=1).item()]


model = models.efficientnet_b0()
model.classifier = nn.Sequential(
    nn.Dropout(), nn.Linear(in_features=1280, out_features=10)
)

model.load_state_dict(torch.load("model0.pth", map_location=torch.device("cpu")))


st.title("Image classification with Streamlit")

uploaded_file = st.file_uploader("Upload an image", type="jpg")

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
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)
    col1.image(image, caption="Uploaded image", use_column_width=True)

    predictions = predict(
        model=model,
        image=image,
        class_names=class_names,
    )
    col2.header("Predictions")
    col2.subheader(f"{predictions}")
