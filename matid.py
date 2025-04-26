import os

if not os.path.exists("best.pt"):
    import urllib.request
    url = "https://drive.google.com/uc?id=1ETP6JcrGr5qGATz3Y3xNk_fy7OKinDTx"
    urllib.request.urlretrieve(url, "best.pt")

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from ultralytics import YOLO
from PIL import Image

from material_info import material_data

# CSS styling
st.markdown("""
<style>
html, body, [class*="css"] {
  background-color: #fefae0 !important;
  color: #2f2f2f !important;
  font-family: 'Segoe UI', sans-serif !important;
}
.grid-box {
  background-color: #fff4a3 !important;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 0 4px rgba(0,0,0,0.1);
  margin-bottom: 1.5em;
  min-height: 180px;
  color: #2f2f2f;
  font-size: 16px;
  line-height: 1.4;
}
.grid-title {
  font-weight: 600;
  font-size: 18px;
  margin-bottom: 8px;
}
.nav-buttons {
  display: flex;
  justify-content: space-between;
  margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = YOLO("best.pt")

label_map = {
    "plastic_bottle": "plastic",
    "metal_can": "metal",
    "cardboard_box": "cardboard",
    "biodegradable_trash": "biodegradable trash",
    "biodegradable": "biodegradable trash",
    "glass_jar": "glass",
    "plastic": "plastic",
    "metal": "metal",
    "cardboard": "cardboard",
    "glass": "glass"
}

def display_material_info(key):
    info = material_data.get(key)
    if not info:
        st.warning(f"No info available for: {key}")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='grid-box'>
            <div class='grid-title'>🧱 Material & Recyclability</div>
            <b>{key.capitalize()}</b><br>{info['description']}<br><br>
            <b>Recyclable:</b> {'✅' if info['recyclable'] else '❌'}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='grid-box'>
            <div class='grid-title'>🌍 Environmental Impact</div>
            {info['impact']}
        </div>
        """, unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        steps = '<br>'.join(f"{i+1}. {s}" for i, s in enumerate(info['how_to_dispose']))
        st.markdown(f"""
        <div class='grid-box'>
            <div class='grid-title'>📦 How to Dispose</div>
            {steps}
        </div>
        """, unsafe_allow_html=True)
    with c4:
        bin_path = os.path.join('image', os.path.basename(info['bin_image']))
        if os.path.exists(bin_path):
            st.image(bin_path, caption='Place this in the appropriate bin.', use_container_width=True)
        else:
            st.warning(f"Bin image not found: {bin_path}")

# Centered title
st.markdown(
    """
    <h1 style='text-align:center; font-family: "Segoe UI", sans-serif; font-size:48px; margin-bottom:0.5em;'>
    MatID
    </h1>
    """, unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    max_idx = len(uploaded_files) - 1

    cols = st.columns([1, 2, 1])
    with cols[0]:
        if st.button("← Previous") and st.session_state.idx > 0:
            st.session_state.idx -= 1
    with cols[2]:
        if st.button("Next →") and st.session_state.idx < max_idx:
            st.session_state.idx += 1

    current = uploaded_files[st.session_state.idx]
    st.subheader(f"{current.name} ({st.session_state.idx+1}/{len(uploaded_files)})")
    img = Image.open(current).convert("RGB")

    prediction_results = model.predict(img, imgsz=416, conf=0.2, stream=True)
    res = next(prediction_results)

    # Display original image first
    st.image(img, caption="Original Image", use_container_width=True)

    # Then display detection result
    result_img = res.plot()
    st.image(result_img, caption="Detected Materials", use_container_width=True)

    st.markdown("**Detection Confidence**")
    labels = set()

    for box in res.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        key = label_map.get(cls_name, cls_name)
        labels.add(key)
        conf = float(box.conf[0])
        st.write(f"{key.capitalize()}: {conf * 100:.1f}%")
        st.progress(conf)

    st.markdown("**Material Information**")
    for label in labels:
        display_material_info(label)
