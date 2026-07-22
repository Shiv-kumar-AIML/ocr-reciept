import streamlit as st
import pandas as pd
import requests
import base64
import json
from PIL import Image, ImageEnhance
import io

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(
    page_title="Receipt OCR",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Receipt Product Extractor (GLM-OCR + Ollama)")

st.write("Upload a receipt image to extract product details.")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "glm-ocr:latest"

receipt_format = st.radio(
    "Select Receipt Format (Crucial for correct Barcodes):",
    [
        "Format A: Barcode ABOVE Product Name (e.g. Lulu)", 
        "Format B: Product Name ABOVE Barcode (e.g. Nesto)"
    ]
)

if "Barcode ABOVE" in receipt_format:
    format_instruction = "CRITICAL RULE: The barcode is ALWAYS printed ABOVE the product name. You MUST match the English product name to the barcode directly ABOVE it."
    format_example = """Example (Barcode on TOP):
*1234567890123     3 x     5.00   S2
Arabic Text Here
Sample Coffee Product    15.00
JSON: {"product_name": "Sample Coffee Product", "barcode": "1234567890123", "quantity": "3", "price": "5.00", "total_amount": "15.00"}"""
else:
    format_instruction = "CRITICAL RULE: The English product name is ALWAYS printed ABOVE the barcode. You MUST match the English product name to the barcode directly BELOW it."
    format_example = """Example (Name on TOP):
Sample Weight Item 0.88   19.99  17.59
Arabic Text Here
9876543210123
JSON: {"product_name": "Sample Weight Item", "barcode": "9876543210123", "quantity": "0.88", "price": "19.99", "total_amount": "17.59"}"""

uploaded_file = st.file_uploader(
    "Upload Receipt Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.image(uploaded_file, width="stretch")

    if st.button("Extract Products"):

        with st.spinner("Reading receipt..."):

            image_bytes = uploaded_file.read()
            
            # Resize and enhance image to improve OCR accuracy
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((1600, 1600))  # Reverted back to 1600 to avoid 500 Error
            
            # Enhance image (Sharpness and Contrast)
            img = ImageEnhance.Sharpness(img).enhance(2.0)
            img = ImageEnhance.Contrast(img).enhance(1.5)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            image_bytes = buffered.getvalue()

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            prompt = f"""
You are an OCR receipt extraction AI.

Extract every purchased product.

Return ONLY valid JSON.

Schema:

{{
  "products":[
    {{
      "product_name":"",
      "barcode":"",
      "quantity":"",
      "price":"",
      "total_amount":""
    }}
  ]
}}

Rules:

1. Do not explain. Do not add markdown. Return JSON only.
2. {format_instruction}
3. Extract the English product name, barcode (without asterisks), quantity, and unit price.

Examples of correct extraction (Do NOT copy these exact values, use them only to understand the format):

{format_example}

Extract ALL products from the provided image using this logic.
"""

            payload = {
                "model": MODEL_NAME,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [
                            image_base64
                        ]
                    }
                ]
            }

            try:

                response = requests.post(
                    OLLAMA_URL,
                    json=payload,
                    timeout=300
                )

                response.raise_for_status()

                data = response.json()

                result_text = data["message"]["content"]

                # Remove markdown if present
                result_text = result_text.replace("```json", "")
                result_text = result_text.replace("```", "")
                result_text = result_text.strip()

                result = json.loads(result_text)

                products = result.get("products", [])

                if len(products) == 0:
                    st.warning("No products found.")
                else:

                    df = pd.DataFrame(products)

                    st.success(f"{len(df)} Products Found")

                    st.dataframe(
                        df,
                        width="stretch"
                    )

                    csv = df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name="receipt_products.csv",
                        mime="text/csv"
                    )

                    with st.expander("JSON Output"):
                        st.json(result)

            except json.JSONDecodeError:

                st.error("Model did not return valid JSON.")

                st.code(result_text)

            except Exception as e:

                st.error(str(e))