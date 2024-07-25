import base64
import json
import os

import requests
import streamlit as st
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

categories = {
    "women": {
        "accessories": [
            "glasses",
            "watches",
            "jewelry & costume jewelry",
            "scarves",
            "belts",
            "headscarves",
            "fans",
            "gloves",
            "ponchos",
            "beanies",
            "hats",
            "hair accessories",
        ],
        "clothes": [
            "blouses",
            "vests",
            "pants",
            "shirts",
            "coats & jackets",
            "jumpsuits",
            "skirts",
            "shorts & bermudas",
            "suits",
            "dresses",
            "kimonos",
            "beach",
            "intimate fashion",
            "pajamas",
            "socks",
            "costumes",
        ],
        "beauty": ["makeup", "skincare", "perfumes", "hair", "nails", "accessories"],
        "shoes": [
            "boots",
            "sandals & flats",
            "ballet flats",
            "shoes",
            "sneakers",
            "flip-flops",
            "mules",
            "slippers",
            "accessories",
        ],
        "bags": [
            "trunk",
            "briefcase",
            "bucket bag",
            "wallet & document holder",
            "messenger bag",
            "clutch",
            "crossbody",
            "sports bag",
            "fan",
            "suitcase",
            "handbag",
            "messenger bag",
            "mini bag",
            "backpack",
            "toiletry bag",
            "shoulder bag",
            "parts & accessories",
            "fanny pack",
            "satchel",
            "crossbody bag",
            "tote",
            "others",
        ],
        "other": [],
    },
    "men": {
        "accessories": [
            "glasses",
            "watches",
            "hats",
            "belts",
            "ties",
            "scarves",
            "scarf",
            "ponchos",
            "wallets",
            "beanies",
            "gloves",
            "jewelry & costume jewelry",
            "hair accessories",
        ],
        "beauty": ["perfumes", "skincare", "hair"],
        "shoes": [
            "boots",
            "sneakers",
            "sandals & flats",
            "shoes",
            "flip-flops",
            "mules",
            "slippers",
            "accessories",
        ],
        "clothing": [
            "shorts & bermudas",
            "suits",
            "coats & jackets",
            "pants",
            "shirts",
            "blouses",
            "vests",
            "jumpsuits",
            "kimonos",
            "pajamas",
            "socks",
            "underwear",
            "beachwear",
            "costumes",
        ],
        "bags": [
            "fanny pack",
            "messenger bag",
            "briefcase",
            "backpack",
            "suitcase",
            "sports bag",
            "parts & accessories",
        ],
        "other": [],
    },
}


def get_index(department, category, subcategory, categories=categories):
    department_index = list(categories.keys()).index(department)
    category_index = list(categories[department].keys()).index(category)
    subcategory_index = categories[department][category].index(subcategory)

    return department_index, category_index, subcategory_index


predictions = {
    "title": "",
    "description": "",
    "department": "men",
    "category": "accessories",
    "subcategory": "glasses",
}


def analyze_image(image_b64, categories: dict = categories):
    system_prompt = """
    You are an agent specialized in analyzing images of products to create an
    appropriate title, description, and tag them with keywords that could be
    used to create an ad for these items on a marketplace.

    You will be provided with:
    - an image
    - a dictionary of possible departments, their possible categories, and their sub-categories

    Your goal is then
    - write a short informal title for the marketplace ad
    - write a informal description for the item, with up to 350 characters
    - to pick a department that best describe the item
    - to pick a category and a sub-category that best describe the item

    Return the keywords in json format, like this:
    {
        'title': 'brand summer dress',
        'description': 'Vintage floral dress, size S. Good condition with minor wear. Great for retro-themed parties!',
        'department': 'women',
        'category': 'clothes',
        'subcategory': 'blouses',
    }
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
            {
                "role": "user",
                "content": f"Categories: {', '.join(categories)}",
            },
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    response_data = response.json()
    return json.loads(response_data["choices"][0]["message"]["content"])


st.title("Image classification with Streamlit")

col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("Upload an image", type="jpg")

for key, value in predictions.items():
    if key not in st.session_state:
        st.session_state[key] = value  # default index

department_index, category_index, subcategory_index = None, None, None
if uploaded_file is not None:
    col1.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    # Convert the uploaded file to bytes
    image_bytes = uploaded_file.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    predictions = analyze_image(image_b64)
    st.write(predictions)

    # department_index, category_index, subcategory_index = get_index(
    #     predictions["department"], predictions["category"], predictions["subcategory"]
    # )

    department_index, category_index, subcategory_index = 0, 2, 3
    st.write(department_index, category_index, subcategory_index)
    for key, value in predictions.items():
        st.session_state[key] = predictions[key]

col2.header("Product")

col2.text_input(
    value=st.session_state["title"],
    label="Title",
    placeholder="eg: 'summer dress' or 'running shirt'",
    max_chars=None,
)

col2.text_input(
    label="brand", placeholder="What's the product's brand?", max_chars=None
)
col2.checkbox(label="No brand", value=False)

col2.text_area(
    value=st.session_state["description"],
    label="description",
    placeholder="eg: 'Vintage floral dress, size S. Good condition with minor wear. Great for retro-themed parties!'",
)

col2.checkbox(label="Used?", value=True)


selected_department = col2.selectbox(
    label="Department",
    options=categories.keys(),
    index=department_index,
    label_visibility="visible",
)

selected_category = col2.selectbox(
    label="Category",
    options=categories.get(selected_department, {}).keys(),
    index=category_index,
    label_visibility="visible",
)

col2.selectbox(
    label="Sub-Category",
    options=categories.get(selected_department, {}).get(selected_category, {}),
    index=subcategory_index,
    label_visibility="visible",
)
