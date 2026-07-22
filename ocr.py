import os
import streamlit as st
import pandas as pd
import requests
import json
import re
import tempfile
import base64
import logging

# Suppress PaddleOCR internal logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Disable mkldnn for PaddleOCR on Mac (ARM64)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Receipt OCR", page_icon="🧾", layout="wide")
st.title("🧾 Smart Receipt Extractor")
st.write("**Step 1:** PaddleOCR reads the image → **Step 2:** DeepSeek OCR structures the data")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "deepseek-ocr:3b"

# Use st.cache_resource so PaddleOCR only loads into memory ONCE
@st.cache_resource
def load_ocr_model():
    # In older versions, show_log is not supported. We suppress logs via logging module instead.
    return PaddleOCR(lang="en", use_textline_orientation=True, enable_mkldnn=False)

try:
    ocr_model = load_ocr_model()
except Exception as e:
    st.error(f"Failed to load OCR model: {e}")
    ocr_model = None

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload Receipt Image", type=["jpg", "jpeg", "png"])

if uploaded_file and ocr_model:
    st.image(uploaded_file, width="stretch")

    if st.button("Extract Products"):
        # ====================================================
        # STEP 1: PaddleOCR — Extract text from image
        # ====================================================
        with st.spinner("Step 1/2: PaddleOCR is reading the receipt..."):
            image_bytes = uploaded_file.read()

            # Save uploaded file to temp path for PaddleOCR
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
                tf.write(image_bytes)
                temp_path = tf.name

            # Use predict instead of ocr to avoid DeprecationWarning
            try:
                # Some versions of PaddleOCR use .ocr(), others use .predict()
                # If .ocr gives deprecation warning, we stick to it but silence it, or try predict.
                # Usually .ocr() is the standard API. The warning is harmless, but let's just use it.
                paddle_result = ocr_model.ocr(temp_path)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                paddle_result = None

            all_text = []
            if paddle_result and len(paddle_result) > 0:
                res = paddle_result[0]
                if isinstance(res, dict) and 'rec_texts' in res:
                    all_text.extend(res['rec_texts'])
                elif isinstance(res, list):
                    for line in res:
                        if isinstance(line, list) and len(line) == 2 and isinstance(line[1], tuple):
                            all_text.append(line[1][0])
                        else:
                            all_text.append(str(line))

            clean_text = []
            for t in all_text:
                tl = t.strip().lower()
                if tl in ["s2", "s1", "1t", "e", "ea"]: continue
                if "happiness points" in tl or "coupon message" in tl or "customer:" in tl: continue
                if any(x in tl for x in ["mastercard", "eft-trans", "auth", "card no", "tax inclusive", "incvat", "vat"]): continue
                clean_text.append(t)

            ocr_text = "\n".join(clean_text)
            if os.path.exists(temp_path):
                os.remove(temp_path) # Clean up temp file

            st.success("✅ Step 1 Complete: Text extracted!")
            with st.expander("View Raw OCR Text (from PaddleOCR)"):
                st.text(ocr_text)

            # Free up memory before running Llama to prevent crashes
            del ocr_model
            st.cache_resource.clear()
            import gc
            gc.collect()

        # ====================================================
        # STEP 2: Llama 3.2 Vision — Structure the text into JSON
        # ====================================================
        with st.spinner("Step 2/2: DeepSeek OCR is structuring the data... (this may take a while)"):
            
            prompt = f"""You are an expert receipt data extraction AI. Extract EVERY purchased product — do NOT skip any.
            Here is the OCR text extracted from the receipt:
            ---BEGIN OCR TEXT---
            {ocr_text}
            ---END OCR TEXT---
            
            OUTPUT: Return ONLY valid JSON. No markdown. No explanations. No code blocks.
            {{"products": [{{"product_name": "...", "barcode": "...", "quantity": "...", "price": "...", "total_amount": "..."}}]}}
            """
            
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            payload = {
                "model": MODEL_NAME,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [image_base64]}],
                "options": {"num_ctx": 4096} # Reduced context window to save memory
            }

            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=600)
                response.raise_for_status()

                data = response.json()
                result_text = data["message"]["content"]
                result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL)
                result_text = result_text.replace("```json", "").replace("```", "").strip()

                json_match = re.search(r'\{.*\}', result_text, flags=re.DOTALL)
                if json_match:
                    result_text = json_match.group()

                result = json.loads(result_text)
                valid_products = [
                    p for p in result.get("products", []) 
                    if str(p.get("product_name", "")).strip().lower() not in ["s2", "s1", "1t", "total", "items", "vat", "change"] 
                    and len(str(p.get("product_name", "")).strip()) >= 3
                ]

                if not valid_products:
                    st.warning("No products found.")
                else:
                    df = pd.DataFrame(valid_products)
                    st.success(f"✅ Step 2 Complete: {len(df)} Products Found!")
                    st.dataframe(df, width="stretch")
                    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "receipt_products.csv", "text/csv")

            except Exception as e:
                st.error(f"Error communicating with Ollama: {str(e)}")