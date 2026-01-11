import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Page configuration
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

st.title("🔷 Shape & Contour Analyzer")
st.write("Upload an image to detect shapes, count objects, and calculate area & perimeter.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    sides = len(approx)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        return "Square / Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    elif sides > 6:
        return "Circle"
    else:
        return "Unknown"

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Preprocessing
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaModify blur? no.
GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img_cv.copy()
    records = []

    obj_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # Noise filtering
            continue

        shape = classify_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Draw contour
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(output, shape, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        records.append([
            obj_id,
            shape,
            round(area, 2),
            round(perimeter, 2)
        ])
        obj_id += 1

    # Convert image for display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Shapes")
        st.image(output_rgb, use_column_width=True)

    with col2:
        st.subheader("Shape Details")
        if records:
            df = pd.DataFrame(
                records,
                columns=["Object ID", "Shape", "Area (pixels)", "Perimeter (pixels)"]
            )
            st.dataframe(df)
            st.success(f"Total Objects Detected: {len(df)}")
        else:
            st.warning("No shapes detected.")

else:
    st.info("Please upload an image to start analysis.")
