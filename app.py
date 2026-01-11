import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page setup
st.set_page_config(
    page_title="Geometric Shape Detector",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎨 Geometric Shape Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Computer Vision Shape Analysis Tool</p>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Control Panel")
    st.markdown("---")
    
    # Detection settings
    st.subheader("Detection Settings")
    blur_kernel = st.select_slider("Blur Intensity", options=[3, 5, 7, 9, 11], value=5)
    threshold_method = st.selectbox("Threshold Method", ["Binary", "Adaptive Mean", "Adaptive Gaussian"])
    threshold_val = st.slider("Threshold Value", 50, 200, 120)
    
    st.markdown("---")
    
    # Filter settings
    st.subheader("Object Filters")
    min_contour_area = st.number_input("Min Area (px²)", 100, 10000, 800, 100)
    max_contour_area = st.number_input("Max Area (px²)", 1000, 100000, 50000, 1000)
    approx_precision = st.slider("Detection Precision", 0.01, 0.10, 0.035, 0.005)
    
    st.markdown("---")
    
    # Display options
    st.subheader("Display Options")
    show_labels = st.checkbox("Show Shape Labels", True)
    show_measurements = st.checkbox("Show Measurements", True)
    show_centers = st.checkbox("Show Center Points", False)

def identify_shape(contour, precision):
    """Enhanced shape identification with more details"""
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, precision * perimeter, True)
    vertex_count = len(approximation)
    
    # Get additional properties
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(approximation)
    aspect_ratio = w / float(h) if h != 0 else 0
    
    # Determine shape type
    if vertex_count == 3:
        return "Triangle", vertex_count, (255, 50, 50)
    elif vertex_count == 4:
        if 0.90 <= aspect_ratio <= 1.10:
            return "Square", vertex_count, (50, 255, 50)
        else:
            return "Rectangle", vertex_count, (50, 50, 255)
    elif vertex_count == 5:
        return "Pentagon", vertex_count, (255, 255, 50)
    elif vertex_count == 6:
        return "Hexagon", vertex_count, (255, 50, 255)
    elif vertex_count == 7:
        return "Heptagon", vertex_count, (50, 255, 255)
    elif vertex_count == 8:
        return "Octagon", vertex_count, (255, 150, 50)
    elif vertex_count > 8:
        # Check circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        if circularity > 0.8:
            return "Circle", vertex_count, (150, 50, 255)
        else:
            return "Ellipse", vertex_count, (100, 200, 100)
    else:
        return "Unknown", vertex_count, (128, 128, 128)

def analyze_image(image, blur_k, thresh_method, thresh_val, min_area, max_area, precision):
    """Complete image analysis pipeline"""
    # Convert to array
    img_array = np.array(image)
    original_img = img_array.copy()
    
    # Preprocessing
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply blur
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    # Apply threshold based on method
    if thresh_method == "Binary":
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    elif thresh_method == "Adaptive Mean":
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    else:  # Adaptive Gaussian
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Annotated image
    annotated = original_img.copy()
    
    # Results storage
    results = []
    shape_id = 1
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(contour, True)
            shape_name, vertices, color = identify_shape(contour, precision)
            
            # Get moments for center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate additional metrics
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Calculate extent
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Draw contour
            cv2.drawContours(annotated, [contour], -1, color, 3)
            
            # Draw label
            if show_labels:
                label = f"{shape_name} #{shape_id}"
                cv2.putText(annotated, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw measurements
            if show_measurements:
                measurement_text = f"A:{int(area)}"
                cv2.putText(annotated, measurement_text, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw center
            if show_centers:
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 0), -1)
                cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)
            
            # Store results
            results.append({
                "ID": shape_id,
                "Shape": shape_name,
                "Vertices": vertices,
                "Area (px²)": round(area, 2),
                "Perimeter (px)": round(perimeter, 2),
                "Center X": cx,
                "Center Y": cy,
                "Width": w,
                "Height": h,
                "Solidity": round(solidity, 3),
                "Extent": round(extent, 3)
            })
            
            shape_id += 1
    
    return annotated, binary, blurred, results

# File upload
uploaded = st.file_uploader("📁 Upload Image for Analysis", 
                            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF")

if uploaded is not None:
    # Load and display original
    img = Image.open(uploaded)
    
    # Process image
    with st.spinner("🔍 Analyzing image..."):
        annotated_img, binary_img, blur_img, shape_results = analyze_image(
            img, blur_kernel, threshold_method, threshold_val,
            min_contour_area, max_contour_area, approx_precision
        )
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🖼️ Processing Steps", "📈 Statistics", "📥 Export"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(img, use_container_width=True)
        
        with col2:
            st.subheader("Detected Shapes")
            st.image(annotated_img, use_container_width=True)
        
        if shape_results:
            st.success(f"✅ Successfully detected {len(shape_results)} shapes!")
            
            # Quick metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Shapes", len(shape_results))
            with metric_cols[1]:
                total_area = sum([s["Area (px²)"] for s in shape_results])
                st.metric("Total Area", f"{total_area:.0f} px²")
            with metric_cols[2]:
                avg_perimeter = np.mean([s["Perimeter (px)"] for s in shape_results])
                st.metric("Avg Perimeter", f"{avg_perimeter:.1f} px")
            with metric_cols[3]:
                shapes_list = [s["Shape"] for s in shape_results]
                most_common = max(set(shapes_list), key=shapes_list.count)
                st.metric("Most Common", most_common)
        else:
            st.warning("⚠️ No shapes detected. Try adjusting the parameters.")
    
    with tab2:
        st.subheader("Image Processing Pipeline")
        
        process_cols = st.columns(3)
        
        with process_cols[0]:
            st.markdown("**Step 1: Blur**")
            st.image(blur_img, use_container_width=True, channels="GRAY")
            st.caption(f"Gaussian Blur (kernel: {blur_kernel}x{blur_kernel})")
        
        with process_cols[1]:
            st.markdown("**Step 2: Threshold**")
            st.image(binary_img, use_container_width=True, channels="GRAY")
            st.caption(f"Method: {threshold_method}")
        
        with process_cols[2]:
            st.markdown("**Step 3: Detection**")
            st.image(annotated_img, use_container_width=True)
            st.caption("Contour Detection & Classification")
    
    with tab3:
        if shape_results:
            df = pd.DataFrame(shape_results)
            
            st.subheader("📋 Detailed Shape Information")
            st.dataframe(df, use_container_width=True, height=300)
            
            # Visualizations
            st.subheader("📊 Visual Analytics")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Shape distribution pie chart
                shape_counts = df["Shape"].value_counts().reset_index()
                shape_counts.columns = ["Shape", "Count"]
                
                fig1 = px.pie(shape_counts, values="Count", names="Shape",
                             title="Shape Distribution",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig1, use_container_width=True)
            
            with viz_col2:
                # Area comparison bar chart
                fig2 = px.bar(df.head(10), x="ID", y="Area (px²)",
                             color="Shape", title="Area Comparison (Top 10)",
                             color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Scatter plot
            st.subheader("🎯 Shape Positions")
            fig3 = px.scatter(df, x="Center X", y="Center Y", 
                             size="Area (px²)", color="Shape",
                             hover_data=["ID", "Perimeter (px)"],
                             title="Shape Locations in Image",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            fig3.update_yaxes(autorange="reversed")  # Flip Y-axis to match image coordinates
            st.plotly_chart(fig3, use_container_width=True)
            
            # Statistical summary
            st.subheader("📈 Statistical Summary")
            summary_cols = st.columns(2)
            
            with summary_cols[0]:
                st.markdown("**Area Statistics**")
                st.write(df[["Area (px²)"]].describe())
            
            with summary_cols[1]:
                st.markdown("**Perimeter Statistics**")
                st.write(df[["Perimeter (px)"]].describe())
    
    with tab4:
        if shape_results:
            st.subheader("💾 Export Options")
            
            # CSV export
            csv_data = pd.DataFrame(shape_results).to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"shape_analysis_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON export
                import json
                json_data = json.dumps(shape_results, indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"shape_analysis_{timestamp}.json",
                    mime="application/json"
                )
            
            # Summary report
            st.markdown("---")
            st.subheader("📄 Analysis Summary")
            
            summary = f"""
            **Analysis Report**
            - Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            - Total Shapes Detected: {len(shape_results)}
            - Threshold Method: {threshold_method}
            - Total Area Coverage: {sum([s['Area (px²)'] for s in shape_results]):.2f} px²
            
            **Shape Breakdown:**
            """
            
            shape_counts = pd.DataFrame(shape_results)["Shape"].value_counts()
            for shape, count in shape_counts.items():
                summary += f"\n- {shape}: {count}"
            
            st.text_area("Report", summary, height=300)
            
            st.download_button(
                label="📥 Download Report",
                data=summary,
                file_name=f"analysis_report_{timestamp}.txt",
                mime="text/plain"
            )

else:
    # Welcome screen
    st.info("👆 Upload an image to begin shape detection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - Detect 9+ shape types
        - Real-time parameter tuning
        - Multiple threshold methods
        - Advanced metrics
        """)
    
    with col2:
        st.markdown("### 📊 Analytics")
        st.markdown("""
        - Interactive charts
        - Position mapping
        - Statistical analysis
        - Export to CSV/JSON
        """)
    
    with col3:
        st.markdown("### 🔧 Supported Shapes")
        st.markdown("""
        - Triangles, Squares
        - Rectangles, Pentagons
        - Hexagons, Heptagons
        - Octagons, Circles
        - Ellipses
        """)

# Footer
st.markdown("---")
st.markdown("**🚀 Geometric Shape Detector** | Powered by OpenCV, Streamlit & Plotly")
