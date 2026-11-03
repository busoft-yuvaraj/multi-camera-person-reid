"""
Streamlit Physical Calibration & Ground-Plane Mapping Dashboard
Phase 2 Single-Camera Video Calibration Testing

Features:
- Single-camera video selection or user upload
- Automatic loading of predefined backend floor plans and homographies (no manual upload required)
- YOLO11 person detection + ByteTrack / BoT-SORT tracking
- Image footpoint to physical ground-plane coordinate projection
- Common building coordinate mapping & obstacle gating
- Synchronized dual-panel live rendering (Left: CCTV video, Right: Ground plane trajectory)
- Video export to calibration_result/
- JSON observation export to outputs/tracking_logs.json
"""

import os
import sys
import time
import json
import tempfile
from typing import Optional, Dict, Any

import cv2
import numpy as np
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.calibration.world_coordinate_mapper import WorldCoordinateMapper
from app.calibration.single_camera_processor import SingleCameraCalibrationProcessor

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Boarhead CCTV - Physical Calibration & Ground-Plane Mapping",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich dark-mode aesthetics
st.markdown("""
<style>
    /* Dark luxury palette */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header card */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    .main-header h1 {
        color: #38bdf8;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    /* Panel frames */
    .panel-box {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px;
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status badge */
    .badge-healthy {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #10b981;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "processor" not in st.session_state:
    st.session_state.processor = SingleCameraCalibrationProcessor(
        config_path="config/camera_calibration.yaml",
        model_name="yolo11n.pt",
        output_dir="calibration_result",
        logs_dir="outputs"
    )

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "paused" not in st.session_state:
    st.session_state.paused = False

if "selected_track_filter" not in st.session_state:
    st.session_state.selected_track_filter = "All Tracks"


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎯 Physical Calibration & Ground-Plane Mapping</h1>
    <p>Phase 2: Single-Camera Physical Homography & Metric Ground-Plane Pedestrian Tracking</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Camera & Input Configuration
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 📷 Camera & Calibration")

camera_options = {
    "Passage 1 (CAM A)": "passage1",
    "Pantry (CAM B)": "pantry",
    "Waiting Lobby (CAM C)": "waiting_lobby"
}

selected_cam_label = st.sidebar.selectbox("Select Target Camera", list(camera_options.keys()))
camera_id = camera_options[selected_cam_label]

mapper = st.session_state.processor.mapper
cam_data = mapper.cameras.get(camera_id)
cal_res = mapper.calibration_results.get(camera_id)

if cam_data and cal_res:
    st.sidebar.markdown(f"""
    **Zone:** `{cam_data.zone}`  
    **Orientation:** {cam_data.camera_orientation}  
    **Homography Status:** <span class="badge-healthy">VALID (RMS: {cal_res.rms_reprojection_error:.2f} px)</span>  
    **Condition Number:** `{cal_res.condition_number:.2e}`  
    **Landmarks Configured:** `{len(cam_data.landmarks)}` points
    """, unsafe_allow_html=True)

    # Show predefined floor plan preview in sidebar (NO MANUAL UPLOAD REQUIRED)
    if os.path.exists(cam_data.floor_plan_image):
        st.sidebar.image(
            cam_data.floor_plan_image,
            caption=f"Predefined Floor Plan: {cam_data.zone.upper()}",
            use_container_width=True
        )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎬 Video Source")

video_source_type = st.sidebar.radio("Choose Input Mode", ["Sample CCTV Video", "Upload CCTV File"])

video_file_path = None
if video_source_type == "Sample CCTV Video":
    sample_file = f"cctv_samples/{camera_id}.mp4"
    if os.path.exists(sample_file):
        video_file_path = sample_file
        st.sidebar.success(f"Loaded: `{os.path.basename(sample_file)}`")
    else:
        st.sidebar.error(f"Sample video not found at {sample_file}")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CCTV Video (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov"]
    )
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"uploaded_{camera_id}_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_file_path = temp_path
        st.sidebar.success(f"Uploaded: `{uploaded_file.name}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Tracking Parameters")
conf_threshold = st.sidebar.slider("YOLO11 Confidence", min_value=0.20, max_value=0.90, value=0.35, step=0.05)
tracker_choice = st.sidebar.selectbox("Local Tracker", ["bytetrack.yaml", "botsort.yaml"])
max_frames_to_run = st.sidebar.slider("Max Frames to Process", min_value=30, max_value=600, value=150, step=30)
start_frame = st.sidebar.number_input("Start Frame Offset", min_value=0, max_value=1500, value=0 if camera_id != "waiting_lobby" else 200, step=25)


# -----------------------------------------------------------------------------
# Control Buttons
# -----------------------------------------------------------------------------
col_b1, col_b2, col_b3 = st.sidebar.columns(3)

start_clicked = col_b1.button("▶ Start", use_container_width=True, type="primary")
pause_clicked = col_b2.button("⏸ Pause", use_container_width=True)
reset_clicked = col_b3.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    st.session_state.processor.reset_trajectories()
    st.session_state.is_processing = False
    st.success("Trajectories and tracking logs reset successfully.")

if pause_clicked:
    st.session_state.paused = not st.session_state.paused


# -----------------------------------------------------------------------------
# Main Dashboard Layout
# -----------------------------------------------------------------------------

# Metrics Row
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric("Camera Feed", camera_id.upper())
with m_col2:
    st.metric("Homography RMS", f"{cal_res.rms_reprojection_error:.2f} px" if cal_res else "N/A")
with m_col3:
    st.metric("Condition Num", f"{cal_res.condition_number:.1e}" if cal_res else "N/A")
with m_col4:
    track_count = len(st.session_state.processor.trajectories)
    st.metric("Active Tracks", track_count)
with m_col5:
    obs_count = len(st.session_state.processor.tracking_logs)
    st.metric("Total Detections", obs_count)

st.markdown("---")

# Two Synchronized Panels Side-by-Side
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown("#### 📹 CCTV Video Panel (Detection & Footpoints)")
    cctv_placeholder = st.empty()
    # Initial preview if not started
    if video_file_path and os.path.exists(video_file_path):
        cap_prev = cv2.VideoCapture(video_file_path)
        cap_prev.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap_prev.read()
        cap_prev.release()
        if ret:
            cctv_placeholder.image(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

with col_right:
    st.markdown("#### 🗺️ Predefined Ground-Plane Map (Physical Trajectory)")
    gp_placeholder = st.empty()
    initial_gp = st.session_state.processor.get_predefined_ground_plan_canvas(camera_id, (720, 720))
    gp_placeholder.image(cv2.cvtColor(initial_gp, cv2.COLOR_BGR2RGB), use_container_width=True)

progress_bar = st.empty()
status_placeholder = st.empty()


# -----------------------------------------------------------------------------
# Video Processing Execution Loop
# -----------------------------------------------------------------------------
if start_clicked and video_file_path:
    st.session_state.is_processing = True
    st.session_state.processor.reset_trajectories()

    cap = cv2.VideoCapture(video_file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video Writers for calibration_result/
    output_dir = "calibration_result"
    os.makedirs(output_dir, exist_ok=True)
    cctv_out_path = os.path.join(output_dir, f"{camera_id}_live_cctv_processed.mp4")
    gp_out_path = os.path.join(output_dir, f"{camera_id}_ground_plane_trajectory.mp4")
    dual_out_path = os.path.join(output_dir, f"{camera_id}_synchronized_dual_view.mp4")

    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_cctv = cv2.VideoWriter(cctv_out_path, fourcc, fps, (w_in, h_in))
    writer_gp = cv2.VideoWriter(gp_out_path, fourcc, fps, (720, 720))
    writer_dual = cv2.VideoWriter(dual_out_path, fourcc, fps, (1280 + 720, 720))

    frame_counter = 0
    t_start = time.time()

    while cap.isOpened() and frame_counter < max_frames_to_run:
        if st.session_state.paused:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        current_frame_idx = start_frame + frame_counter
        timestamp_sec = current_frame_idx / fps

        # Track filter if selected
        active_filter = None
        if st.session_state.selected_track_filter != "All Tracks":
            try:
                active_filter = int(st.session_state.selected_track_filter.replace("Track #", ""))
            except Exception:
                active_filter = None

        cctv_p, gp_p, dual_p, obs = st.session_state.processor.process_frame(
            camera_id=camera_id,
            frame=frame,
            frame_idx=current_frame_idx,
            timestamp_sec=timestamp_sec,
            conf_threshold=conf_threshold,
            tracker_type=tracker_choice,
            active_track_filter=active_filter
        )

        # Write to files
        writer_cctv.write(cctv_p)
        writer_gp.write(gp_p)
        writer_dual.write(dual_p)

        # Update Streamlit UI (subsample display for high framerate smoothness)
        if frame_counter % 2 == 0:
            cctv_placeholder.image(cv2.cvtColor(cctv_p, cv2.COLOR_BGR2RGB), use_container_width=True)
            gp_placeholder.image(cv2.cvtColor(gp_p, cv2.COLOR_BGR2RGB), use_container_width=True)

            pct = (frame_counter + 1) / max_frames_to_run
            progress_bar.progress(pct)
            status_placeholder.text(f"Processing Frame {current_frame_idx} ({frame_counter + 1}/{max_frames_to_run}) | {fps:.1f} FPS")

        frame_counter += 1

    cap.release()
    writer_cctv.release()
    writer_gp.release()
    writer_dual.release()

    # Save JSON tracking logs
    json_log_path = "outputs/tracking_logs.json"
    with open(json_log_path, "w") as f:
        json.dump({
            "camera_id": camera_id,
            "video_file": os.path.basename(video_file_path),
            "total_frames_processed": frame_counter,
            "processing_fps": round(frame_counter / max(0.001, (time.time() - t_start)), 2),
            "total_observations": len(st.session_state.processor.tracking_logs),
            "unique_tracks": list(st.session_state.processor.trajectories.keys()),
            "observations": st.session_state.processor.tracking_logs
        }, f, indent=2)

    meta_log_path = "outputs/calibration_metadata.json"
    with open(meta_log_path, "w") as f:
        json.dump({
            "camera_id": camera_id,
            "is_calibrated": cal_res.is_valid,
            "rms_reprojection_error_px": cal_res.rms_reprojection_error,
            "max_reprojection_error_px": cal_res.max_reprojection_error,
            "condition_number": cal_res.condition_number,
            "H_ground2img": cal_res.H_ground2img.tolist(),
            "H_img2ground": cal_res.H_img2ground.tolist(),
            "landmarks": cal_res.per_point_errors
        }, f, indent=2)

    progress_bar.progress(1.0)
    st.success(f"Processing Complete! Exported {frame_counter} frames to `calibration_result/` and logs to `outputs/tracking_logs.json`.")
    st.session_state.is_processing = False


# -----------------------------------------------------------------------------
# Artifact Downloads & Export Section
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📥 Generated Artifacts & Video Downloads")

col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

cctv_mp4 = f"calibration_result/{camera_id}_live_cctv_processed.mp4"
gp_mp4 = f"calibration_result/{camera_id}_ground_plane_trajectory.mp4"
dual_mp4 = f"calibration_result/{camera_id}_synchronized_dual_view.mp4"
json_path = "outputs/tracking_logs.json"
meta_path = "outputs/calibration_metadata.json"

with col_d1:
    if os.path.exists(cctv_mp4):
        with open(cctv_mp4, "rb") as f:
            st.download_button("📹 Processed CCTV MP4", f, file_name=f"{camera_id}_cctv_processed.mp4", mime="video/mp4", use_container_width=True)

with col_d2:
    if os.path.exists(gp_mp4):
        with open(gp_mp4, "rb") as f:
            st.download_button("🗺️ Ground-Plane MP4", f, file_name=f"{camera_id}_ground_plane.mp4", mime="video/mp4", use_container_width=True)

with col_d3:
    if os.path.exists(dual_mp4):
        with open(dual_mp4, "rb") as f:
            st.download_button("🎬 Synchronized Dual MP4", f, file_name=f"{camera_id}_dual_synchronized.mp4", mime="video/mp4", use_container_width=True)

with col_d4:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            st.download_button("📄 Tracking Logs JSON", f.read(), file_name="tracking_logs.json", mime="application/json", use_container_width=True)

with col_d5:
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            st.download_button("📐 Calibration Meta JSON", f.read(), file_name="calibration_metadata.json", mime="application/json", use_container_width=True)


# -----------------------------------------------------------------------------
# Tracking Observation Table & Detailed Calibration Inspection
# -----------------------------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Tracking Logs Table", "📐 Calibration Matrices & Landmarks", "🌐 Common Building Coordinates"])

with tab1:
    logs = st.session_state.processor.tracking_logs
    if logs:
        st.markdown(f"**Total Observations Logged:** {len(logs)}")
        table_rows = []
        for o in logs[-25:]:  # show recent 25
            loc_g = o["world_coordinates"]["local_ground"]
            glob_g = o["world_coordinates"]["global_building"]
            table_rows.append({
                "Frame": o["frame_number"],
                "Time (s)": o["timestamp"],
                "Track ID": f"Track #{o['track_id']}",
                "Conf": f"{o['confidence']:.2f}",
                "Footpoint (u, v)": f"({o['footpoint_pixel'][0]}, {o['footpoint_pixel'][1]})",
                "Ground (X, Y)": f"({loc_g[0]:.1f}, {loc_g[1]:.1f})" if loc_g else "None",
                "Building Global": f"({glob_g[0]:.1f}, {glob_g[1]:.1f})" if glob_g else "None",
                "Spatial Status": o["calibration_validity"]["status"],
                "Region": o["calibration_validity"].get("region_name") or "None"
            })
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No tracking observations recorded yet. Press '▶ Start' above to run tracking.")

with tab2:
    if cal_res:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### $H_{ground \\to img}$ (Ground-to-Image Homography)")
            st.code(np.array2string(cal_res.H_ground2img, precision=4, suppress_small=True))
        with c2:
            st.markdown("#### $H_{img \\to ground}$ (Image-to-Ground Homography)")
            st.code(np.array2string(cal_res.H_img2ground, precision=6, suppress_small=True))

        st.markdown("#### Landmark Reprojection Residuals")
        st.dataframe(cal_res.per_point_errors, use_container_width=True)

with tab3:
    if cam_data:
        gt = cam_data.global_transform
        st.markdown(f"""
        **Common Building Coordinate System Reference:**
        - **Translation (Origin in Building Frame):** `({gt.translation.x}, {gt.translation.y})`
        - **Orientation Rotation:** `{gt.rotation_deg}°`
        - **Metric Scaling Factor:** `{gt.scale}`
        - **Topology Connectivity:** Junction at building coordinate `(80.0, 50.0)`
        """)
