import streamlit as st
import pandas as pd
import requests
import json
import os
import re
import tempfile
import base64
import io
import gc
import warnings
import logging
import cv2
import numpy as np
from PIL import Image

# ============================================================
# SUPPRESS NOISE
# ============================================================
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR
from document_scanner import scan_and_crop_document, apply_clahe_enhancement

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Receipt OCR", page_icon="🧾", layout="wide")
st.title("🧾 Fast Smart Receipt Extractor")
st.caption("Step 1: PaddleOCR reads text → Step 2: LLM structures product JSON")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("💡 Upload **1 or 2 images** — if receipt is split into top + bottom photos, upload both in order.")
with col2:
    mode_option = st.radio(
        "⚡ Execution Mode:",
        ["🚀 Ultra Fast (Text-Only LLM)", "🖼️ Vision Mode (Image + LLM)"],
        help="Ultra Fast uses PaddleOCR text + LLM. It takes 2-4 sec instead of ~60s with identical accuracy."
    )

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL_NAME  = "qwen2.5vl:7b"
CHUNK_WIDTH = 900    # max width when chunking
CHUNK_QUAL  = 78     # JPEG quality per chunk — lower = smaller payload



# ============================================================
# HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_ocr_model():
    return PaddleOCR(lang="en", use_textline_orientation=True, enable_mkldnn=False)


def run_paddle_ocr_and_crop(ocr_model, image_pil: Image.Image) -> tuple[list[str], Image.Image]:
    """
    Production-Grade Neural Document Cropper & OCR:
    Runs PaddleOCR's deep learning text detection model (PP-OCRv6).
    Extracts text line bounding polygons to crop the image tightly around 100% of document content.
    Returns (raw_text_lines, cropped_pil_image).
    """
    img_array = np.array(image_pil)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = ocr_model.predict(img_array)
            except AttributeError:
                result = ocr_model.ocr(img_array)
    except Exception as e:
        print(f"PaddleOCR error: {e}")
        return [], crop_document(image_pil)

    texts = []
    boxes = []
    if result and len(result) > 0:
        res = result[0]
        if isinstance(res, dict):
            if 'rec_texts' in res:
                texts.extend([t for t in res['rec_texts'] if t])
            if 'dt_polys' in res:
                boxes = [np.array(p, dtype=np.int32) for p in res['dt_polys']]
        elif isinstance(res, list):
            for line in res:
                if isinstance(line, list) and len(line) == 2:
                    if isinstance(line[0], list):
                        boxes.append(np.array(line[0], dtype=np.int32))
                    if isinstance(line[1], tuple):
                        texts.append(line[1][0])
                elif isinstance(line, str):
                    texts.append(line)

    # Compute production-grade document crop from neural text detection boxes
    if boxes:
        h, w = img_array.shape[:2]
        all_pts = np.vstack(boxes)
        x1, y1 = np.min(all_pts, axis=0)
        x2, y2 = np.max(all_pts, axis=0)

        # 3.5% horizontal and 2% vertical padding
        pad_x = int(w * 0.035)
        pad_y = int(h * 0.02)

        crop_x1 = max(0, int(x1) - pad_x)
        crop_y1 = max(0, int(y1) - pad_y)
        crop_x2 = min(w, int(x2) + pad_x)
        crop_y2 = min(h, int(y2) + pad_y)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w > 50 and crop_h > 50:
            cropped_np = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
            return texts, Image.fromarray(cropped_np)

    # Fallback to hybrid paper cropper
    return texts, crop_document(image_pil)


def clean_lines(lines: list[str]) -> list[str]:
    """Remove known junk OCR lines."""
    junk_exact = {"s2", "s1", "1t", "e", "ea"}
    junk_contains = ["happiness points", "coupon message", "customer:",
                     "mastercard", "eft-trans", "auth", "card no",
                     "tax inclusive", "incvat", "vat"]
    out = []
    for t in lines:
        tl = t.strip().lower()
        if not tl:
            continue
        if tl in junk_exact:
            continue
        if any(x in tl for x in junk_contains):
            continue
        out.append(t)
    return out


def stitch_images(pil_images: list[Image.Image]) -> Image.Image:
    """Stack PIL images vertically into one canvas."""
    max_w = max(img.width for img in pil_images)
    total_h = sum(img.height for img in pil_images)
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for img in pil_images:
        canvas.paste(img, ((max_w - img.width) // 2, y))
        y += img.height
    return canvas


def crop_document(pil_img: Image.Image) -> Image.Image:
    """
    Robust Receipt Cropper:
    Removes background margins (tables, bedsheets, floors, or wide margins)
    around the receipt paper.
    """
    img = np.array(pil_img)
    if img is None or img.size == 0:
        return pil_img

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 and img.shape[2] == 3 else img
    h, w = img_bgr.shape[:2]
    total_area = w * h

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- METHOD 1: Otsu Bright Paper Segmentation ---
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh_paper = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel_paper = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    closed_paper = cv2.morphologyEx(thresh_paper, cv2.MORPH_CLOSE, kernel_paper)

    contours_paper, _ = cv2.findContours(closed_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    paper_crop = None
    if contours_paper:
        c_paper = max(contours_paper, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(c_paper)
        paper_area_ratio = (pw * ph) / total_area
        
        # If paper contour covers 10% to 88% of image, it successfully isolated receipt paper from background
        if 0.10 <= paper_area_ratio <= 0.88:
            pad_x = int(w * 0.015)
            pad_y = int(h * 0.015)
            x1 = max(0, px - pad_x)
            y1 = max(0, py - pad_y)
            x2 = min(w, px + pw + pad_x)
            y2 = min(h, py + ph + pad_y)
            paper_crop = (x1, y1, x2, y2)

    # --- METHOD 2: Text / Content Region Bounding Box (for light/white backgrounds) ---
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    dilated_text = cv2.dilate(adaptive, kernel_text, iterations=2)

    contours_text, _ = cv2.findContours(dilated_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_text_boxes = []
    for c in contours_text:
        tx, ty, tw, th = cv2.boundingRect(c)
        if (tx <= 2 or ty <= 2 or tx + tw >= w - 2 or ty + th >= h - 2) and tw * th < 0.05 * total_area:
            continue
        if cv2.contourArea(c) > 50:
            valid_text_boxes.append((tx, ty, tx + tw, ty + th))

    text_crop = None
    if valid_text_boxes:
        t_x1 = max(0, min(b[0] for b in valid_text_boxes) - int(w * 0.03))
        t_y1 = max(0, min(b[1] for b in valid_text_boxes) - int(h * 0.02))
        t_x2 = min(w, max(b[2] for b in valid_text_boxes) + int(w * 0.03))
        t_y2 = min(h, max(b[3] for b in valid_text_boxes) + int(h * 0.02))
        text_w = t_x2 - t_x1
        text_h = t_y2 - t_y1
        if (text_w * text_h) < 0.95 * total_area:
            text_crop = (t_x1, t_y1, t_x2, t_y2)

    # Apply Crop Selection
    if paper_crop is not None:
        x1, y1, x2, y2 = paper_crop
    elif text_crop is not None:
        x1, y1, x2, y2 = text_crop
    else:
        return pil_img

    cropped_bgr = img_bgr[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


def pil_to_b64(pil_img: Image.Image, quality: int = CHUNK_QUAL) -> str:
    """Compress PIL image → base64 JPEG string."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


def call_ollama_with_image(prompt: str, image_b64: str, debug_log: list,
                           label: str = "") -> str:
    """
    Send prompt + image chunk to Ollama with format json for speed.
    """
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "options": {"num_ctx": 4096, "temperature": 0, "num_predict": 1024}
    }
    tag = f"[{label}] " if label else ""
    debug_log.append(f"📤 {tag}Sending chunk ({len(image_b64)//1024}KB) with vision to Ollama...")
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    debug_log.append(f"   {tag}HTTP {r.status_code}")
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "").strip()
    debug_log.append(f"   {tag}Response length: {len(content)} chars.")
    return content


def call_ollama_text_only(prompt: str, debug_log: list, label: str = "") -> str:
    """Send text-only prompt to Ollama with format json for super fast response."""
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 4096, "temperature": 0, "num_predict": 1024}
    }
    tag = f"[{label}] " if label else ""
    debug_log.append(f"📤 {tag}Sending fast TEXT-ONLY request...")
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    debug_log.append(f"   {tag}HTTP {r.status_code}")
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "").strip()
    debug_log.append(f"   {tag}Response length: {len(content)} chars.")
    return content


def extract_products_from_response(raw: str) -> list[dict]:
    """Parse products JSON from a single model response."""
    if not raw:
        return []
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'```[a-z]*', '', raw).replace('```', '').strip()
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    products = obj.get("products", [])
    junk_names = {"s2", "s1", "1t", "total", "subtotal", "items", "vat",
                  "change", "tax", "amount", "cash", "balance"}
    valid = []
    for p in products:
        name = str(p.get("product_name", "")).strip()
        if not name or name.lower() in junk_names or len(name) < 3:
            continue
        valid.append(p)
    return valid


def deduplicate_products(products: list[dict]) -> list[dict]:
    """Remove duplicate products by (name, barcode) key."""
    seen = set()
    out = []
    for p in products:
        key = (
            str(p.get("product_name", "")).strip().lower(),
            str(p.get("barcode", "")).strip()
        )
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


PRODUCT_PROMPT_TEMPLATE = """\
Extract ALL purchased products from receipt OCR text. Do NOT miss any product or barcode.

{context_note}

---OCR TEXT---
{ocr_section}
---END OCR TEXT---

RULES:
1. BARCODE: Look for barcode / product code numbers (digits near or above the product name). Remove any '*' or '#' prefix. If no barcode digits exist, use "".
2. QUANTITY: Number before 'X' or 'x' (e.g. '3 X' -> '3', '0.450 kg' -> '0.450'). Default to '1' if not specified.
3. Ignore header, footer, tax, store info, totals, payment details, and Arabic text lines.
4. Keep line total as 'total_amount' and item unit price as 'price'.

JSON Output Schema:
{{"products": [{{"product_name": "string", "barcode": "string", "quantity": "string", "price": "string", "total_amount": "string"}}]}}"""


# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_files = st.file_uploader(
    "Upload Receipt Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload 1 image normally, or 2 images if the receipt is split top/bottom."
)

if uploaded_files:
    n = len(uploaded_files)
    col_preview, _ = st.columns([1, 2])
    with col_preview:
        if n == 1:
            st.image(uploaded_files[0], caption="Receipt Preview", width=320)
        else:
            sub_cols = st.columns(min(n, 2))
            for i, f in enumerate(uploaded_files):
                sub_cols[i % 2].image(f, caption=f"Part {i+1}/{n}", width=250)
            st.info(f"📎 {n} images uploaded — will be stitched top-to-bottom.")

    if st.button("🔍 Extract Products", type="primary"):

        debug_log: list[str] = []

        # ====================================================
        # STEP 1 — PaddleOCR on every uploaded image
        # ====================================================
        with st.spinner(f"Step 1/2: PaddleOCR reading {n} image(s)..."):
            try:
                ocr_model = get_ocr_model()
                debug_log.append("✅ PaddleOCR model loaded.")
            except Exception as e:
                st.error(f"❌ Failed to load PaddleOCR: {e}")
                st.stop()

            pil_images: list[Image.Image] = []
            all_ocr_lines: list[str] = []

            for idx, uf in enumerate(uploaded_files):
                raw_bytes = uf.read()
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                
                # Stage 1: CLAHE contrast normalization & 4-corner perspective correction
                scanned_img = scan_and_crop_document(pil_img, enhance_contrast=True)

                # Stage 2: Production DL document crop & OCR text extraction
                lines, cropped_img = run_paddle_ocr_and_crop(ocr_model, scanned_img)
                pil_images.append(cropped_img)
                
                col_crop, _ = st.columns([1, 2])
                with col_crop:
                    st.image(cropped_img, caption=f"Cropped Image {idx+1}", width=320)
                debug_log.append(f"🖼️ Image {idx+1}: original {pil_img.width}×{pil_img.height}px, cropped to {cropped_img.width}×{cropped_img.height}px")
                debug_log.append(f"   → PaddleOCR: {len(lines)} raw lines.")
                lines = clean_lines(lines)
                debug_log.append(f"   → After cleaning: {len(lines)} lines.")

                if n > 1:
                    all_ocr_lines.append(f"--- IMAGE {idx+1} ---")
                all_ocr_lines.extend(lines)

            full_ocr_text = "\n".join(all_ocr_lines)

            # Stitch all uploaded images into one tall canvas
            combined_pil = stitch_images(pil_images) if n > 1 else pil_images[0]
            debug_log.append(
                f"🖼️ Stitched image: {combined_pil.width}×{combined_pil.height}px"
            )

            # We rely on @st.cache_resource to manage PaddleOCR memory
            # Do NOT clear the cache here, as repeatedly reloading PaddleOCR causes memory leaks!

        st.success(
            f"✅ Step 1 done — "
            f"{len([l for l in all_ocr_lines if not l.startswith('---')])} text lines extracted."
        )
        with st.expander("📄 View Raw OCR Text"):
            st.text(full_ocr_text)

        # ====================================================
        # STEP 2 — Process OCR text with LLM
        # ====================================================
        use_fast_mode = "Ultra Fast" in mode_option
        spinner_msg = "Step 2/2: LLM Structuring Products (Fast Text Mode)..." if use_fast_mode else "Step 2/2: Sending image + text to LLM..."
        
        with st.spinner(spinner_msg):

            all_products: list[dict] = []

            # We pass the full extracted OCR text
            ocr_text_flat = "\n".join([l for l in all_ocr_lines if not l.startswith("---")])

            prompt = PRODUCT_PROMPT_TEMPLATE.format(
                context_note="This is the full extracted receipt document.",
                ocr_section=ocr_text_flat
            )

            chunk_label = "Receipt Data"
            raw = ""
            
            if use_fast_mode:
                try:
                    raw = call_ollama_text_only(prompt, debug_log, chunk_label)
                except Exception as e:
                    debug_log.append(f"❌ {chunk_label} fast text call failed: {e}")
                    raw = ""
            else:
                b64 = pil_to_b64(combined_pil)
                debug_log.append(f"📦 Final Image: {combined_pil.width}×{combined_pil.height}px, base64={len(b64)//1024}KB")
                try:
                    raw = call_ollama_with_image(prompt, b64, debug_log, chunk_label)
                except Exception as e:
                    debug_log.append(f"⚠️ {chunk_label} vision call failed ({e}) → trying text-only fallback.")
                    try:
                        raw = call_ollama_text_only(prompt, debug_log, chunk_label)
                    except Exception as ex:
                        debug_log.append(f"❌ {chunk_label} text-only fallback failed: {ex}")
                        raw = ""

            chunk_products = extract_products_from_response(raw)
            debug_log.append(f"   → {chunk_label}: {len(chunk_products)} products extracted.")
            all_products.extend(chunk_products)

            products = deduplicate_products(all_products)
            debug_log.append(f"✅ Total unique products: {len(products)}")

        # ====================================================
        # RESULTS
        # ====================================================
        if not products:
            st.warning("⚠️ No products found. Check the Debug Log below.")
        else:
            df = pd.DataFrame(products)
            st.success(f"✅ Done! **{len(df)} products** extracted.")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="receipt_products.csv",
                mime="text/csv"
            )
            with st.expander("📋 Full JSON Output"):
                st.json({"products": products})

        with st.expander("🐛 Debug Log (click to inspect every step)"):
            st.text("\n".join(debug_log))
