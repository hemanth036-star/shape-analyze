import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

# Page configuration
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Shape & Contour Analyzer")
st.write("Upload an image to detect shapes, count objects, and calculate area & perimeter. This app uses OpenCV for robust contour detection and shape classification.")

# Sidebar for adjustable parameters
st.sidebar.header("Detection Parameters")
min_area = st.sidebar.slider("Minimum Area (pixels)", min_value=100, max_value=5000, value=500, step=100)
canny_low = st.sidebar.slider("Canny Low Threshold", min_value=10, max_value=100, value=50, step=5)
canny_high = st.sidebar.slider("Canny High Threshold", min_value=100, max_value=300, value=150, step=10)
approx_epsilon = st.sidebar.slider("Approximation Epsilon (% of perimeter)", min_value=0.5, max_value=5.0, value=2.0, step=0.1) / 100.0
circularity_threshold = st.sidebar.slider("Circularity Threshold for Circle Detection", min_value=0.7, max_value=1.0, value=0.85, step=0.05)

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)
    sides = len(approx)
    
    # Compute circularity to detect circles
    area = cv2.contourArea(contour)
    if peri > 0:
        circularity = 4 * np.pi * area / (peri * peri)
    else:
        circularity = 0
    
    if circularity > circularity_threshold and sides > 5:
        return "Circle"
    
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        # Check if square or rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    elif sides > 6:
        return "Polygon"
    else:
        return "Unknown"

if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Preprocessing
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection (using adjustable Canny)
        edges = cv2.Canny(blur, canny_low, canny_high)
        
        # Adaptive thresholding for better robustness in varying lighting
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Combine edges and threshold for more robust contour detection
        combined = cv2.bitwise_or(thresh, edges)
        
        # Find contours
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        output = img_cv.copy()
        records = []
        obj_id = 1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:  # Remove noise based on user-defined min area
                continue
            
            shape = classify_shape(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Centroid for labeling
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # Draw contour and label
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(
                output, f"{shape} ({obj_id})", (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )
            
            records.append([
                obj_id,
                shape,
                round(area, 2),
                round(perimeter, 2)
            ])
            obj_id += 1
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detected Shapes")
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            st.image(output_rgb, use_column_width=True)
        
        with col2:
            st.subheader("Shape Details")
            if records:
                df = pd.DataFrame(
                    records,
                    columns=["Object ID", "Shape", "Area (pixels)", "Perimeter (pixels)"]
                )
                st.dataframe(df.style.format({"Area (pixels)": "{:.2f}", "Perimeter (pixels)": "{:.2f}"}))
                st.success(f"Total Objects Detected: {len(df)}")
                
                # Download CSV option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Shape Data as CSV",
                    data=csv,
                    file_name="shape_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No shapes detected. Try adjusting parameters.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an image to start analysis.")
