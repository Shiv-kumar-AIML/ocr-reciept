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


def run_paddle_ocr(ocr_model, image_pil: Image.Image) -> list[str]:
    """Run PaddleOCR on a PIL image. Returns list of text strings."""
    # Convert PIL Image to OpenCV NumPy array to remove tempfile disk buffer
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
        return []

    texts = []
    if not result or len(result) == 0:
        return texts
    res = result[0]
    if isinstance(res, dict) and 'rec_texts' in res:
        texts.extend([t for t in res['rec_texts'] if t])
    elif isinstance(res, list):
        for line in res:
            if isinstance(line, list) and len(line) == 2 and isinstance(line[1], tuple):
                texts.append(line[1][0])
            elif isinstance(line, str):
                texts.append(line)
    return texts


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
    """Detect document contours and apply perspective transform to crop."""
    img = np.array(pil_img)
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is not None:
        pts = doc_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(warped_rgb)
    else:
        return pil_img


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
    if n == 1:
        st.image(uploaded_files[0], caption="Receipt", use_container_width=True)
    else:
        cols = st.columns(n)
        for i, f in enumerate(uploaded_files):
            cols[i].image(f, caption=f"Part {i+1}/{n}", use_container_width=True)
        st.info(f"📎 {n} images uploaded — will be stitched top-to-bottom before processing.")

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
                
                # Crop document before OCR
                cropped_img = crop_document(pil_img)
                pil_images.append(cropped_img)
                st.image(cropped_img, caption=f"Cropped Image {idx+1}", use_container_width=True)
                debug_log.append(f"🖼️ Image {idx+1}: original {pil_img.width}×{pil_img.height}px, cropped to {cropped_img.width}×{cropped_img.height}px")

                lines = run_paddle_ocr(ocr_model, cropped_img)
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
