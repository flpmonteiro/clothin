import base64
import json
import os

import requests
import streamlit as st
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

departments = [
    "women",
    "men",
    "children",
    "other",
]

categories = {
    "women": ["accessories", "bags", "beauty", "clothes", "shoes", "others"],
    "men": ["accessories", "bags", "beauty", "clothes", "shoes", "others"],
    "children": [
        "accessories",
        "bags",
        "clothes",
        "shoes",
        "toys",
        "others",
    ],
    "other": [],
}

subcategories = {
    "accessories": [
        "beanies",
        "belts",
        "fans",
        "glasses",
        "gloves",
        "hair accessories",
        "hats",
        "headscarves",
        "jewelry",
        "ponchos",
        "scarves",
        "ties",
        "wallets",
        "watches",
        "others",
    ],
    "clothes": [
        "beach",
        "blouses",
        "coats & jackets",
        "costumes",
        "dresses",
        "intimate fashion",
        "jumpsuits",
        "kimonos",
        "pajamas",
        "pants",
        "shirts",
        "shorts",
        "skirts",
        "socks",
        "suits",
        "sweaters",
        "vests",
        "others",
    ],
    "beauty": [
        "accessories",
        "hair",
        "makeup",
        "nails",
        "others",
        "perfumes",
        "skincare",
    ],
    "shoes": [
        "ballet flats",
        "boots",
        "flip-flops",
        "sandals & flats",
        "slippers",
        "sneakers",
        "others",
    ],
    "bags": [
        "backpack",
        "briefcase",
        "bucket bag",
        "clutch",
        "crossbody",
        "fan",
        "fanny pack",
        "handbag",
        "messenger bag",
        "mini bag",
        "parts & accessories",
        "satchel",
        "shoulder bag",
        "sports bag",
        "suitcase",
        "toiletry bag",
        "tote",
        "trunk",
        "wallet & document holder",
        "others",
    ],
    "toys": [],
    "others": [],
}


def get_index(
    department,
    category,
    subcategory,
    departments=departments,
    categories=categories,
    subcategories=subcategories,
):
    department_index = departments.index(department)
    category_index = categories.get(department).index(category)
    subcategory_index = subcategories.get(category).index(subcategory)

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
    You are an AI agent trained to analyze product images and generate appropriate marketplace ad content.

    The possible departments are:
    - Men
    - Women
    - Children
    - Other

    The possible categories and their sub-categories are:

    - Accessories: beanies, belts, fans, glasses, gloves, hair accessories, hats, headscarves, jewelry, ponchos, scarves, ties, wallets, watches, others
    - Bags: backpack, briefcase, bucket bag, clutch, crossbody, fan, fanny pack, handbag, messenger bag, mini bag, parts & accessories, satchel, shoulder bag, sports bag, suitcase, toiletry bag, tote, trunk, wallet & document holder, others
    - Beauty: accessories, hair, makeup, nails, others, perfumes, skincare
    - Clothes: beach, blouses, coats & jackets, costumes, dresses, intimate fashion, jumpsuits, kimonos, pajamas, pants, shirts, shorts, skirts, socks, suits, sweaters, vests, others
    - Shoes: ballet flats, boots, flip-flops, sandals & flats, slippers, sneakers, others
    - Toys: (No subcategories)
    - Others: (No subcategories)

    Your tasks are:
    - Create a short, informal ad title
    - Write an informal description of up to 350 characters
    - Select the correct department for the item
    - Select the appropriate category and sub-category

    Return the output in JSON format:
    {
        'title': 'brand summer dress',
        'description': 'Vintage floral dress, size S. Good condition with minor wear. Great for retro-themed parties!',
        'department': 'women',
        'category': 'clothes',
        'subcategory': 'blouses'
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

    department_index, category_index, subcategory_index = get_index(
        predictions["department"], predictions["category"], predictions["subcategory"]
    )

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
    options=categories.get(selected_department, {}),
    index=category_index,
    label_visibility="visible",
)

col2.selectbox(
    label="Sub-Category",
    options=subcategories.get(selected_category, {}),
    index=subcategory_index,
    label_visibility="visible",
)
