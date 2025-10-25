import pandas as pd
import os
from PIL import Image
import numpy as np
from itertools import permutations
import time
import glob
import sys
import streamlit as st
import base64
from io import BytesIO

# ---
# Configuration (Constants)
# ---
GRID_SIZE = 3
N_TILES = GRID_SIZE * GRID_SIZE
IMAGE_SIZE = 225
TILE_SIZE = IMAGE_SIZE // GRID_SIZE
DEFAULT_OUTPUT_FILENAME = "submission.csv"

# ---
# Core Solver Functions (Unchanged from your script)
# ---
def extract_tiles(img):
    """Divides the image into a 3x3 grid of tiles."""
    # Ensure image is resized consistently for processing
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    tiles = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            left = col * TILE_SIZE
            upper = row * TILE_SIZE
            right = left + TILE_SIZE
            lower = upper + TILE_SIZE
            tile = img_resized.crop((left, upper, right, lower))
            tiles.append(tile)
    return tiles # Returns list of 9 PIL Images (75x75)

def edge_difference(tile1, tile2, side1='right', side2='left'):
    """Calculates the difference between adjacent edges of two tiles."""
    t1 = np.array(tile1)
    t2 = np.array(tile2)
    edge1, edge2 = None, None # Initialize

    # Basic check for empty arrays, though shouldn't happen with PIL crop
    if t1.size == 0 or t2.size == 0:
        return float('inf') # Return high difference if tile is invalid

    if side1 == 'right': edge1 = t1[:, -1, :]
    elif side1 == 'left': edge1 = t1[:, 0, :]
    elif side1 == 'top': edge1 = t1[0, :, :]
    elif side1 == 'bottom': edge1 = t1[-1, :, :]
    else: raise ValueError(f"Invalid side1: {side1}")

    if side2 == 'right': edge2 = t2[:, -1, :]
    elif side2 == 'left': edge2 = t2[:, 0, :]
    elif side2 == 'top': edge2 = t2[0, :, :]
    elif side2 == 'bottom': edge2 = t2[-1, :, :]
    else: raise ValueError(f"Invalid side2: {side2}")

    # Ensure edges have compatible shapes before calculating mean
    if edge1.shape != edge2.shape:
        return float('inf') # Mismatched edges = bad fit

    return np.mean((edge1.astype(np.float32) - edge2.astype(np.float32))**2) # Use float32 for calculation

def solve_puzzle(tiles):
    """Finds the best arrangement by minimizing edge differences."""
    n = len(tiles)
    if n != N_TILES:
        print(f"Error: Expected {N_TILES} tiles, got {n}")
        return None # Basic validation

    h_diff = [[float('inf')] * n for _ in range(n)] # Initialize with high diff
    v_diff = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                try: # Add error handling for edge cases
                    h_diff[i][j] = edge_difference(tiles[i], tiles[j], 'right', 'left')
                    v_diff[i][j] = edge_difference(tiles[i], tiles[j], 'bottom', 'top')
                except ValueError as e:
                     print(f"Error in edge_difference for tiles {i},{j}: {e}") # Log error
                     # Keep high diff, don't indicate complete failure yet
                except IndexError as e:
                     print(f"Error accessing edge pixels for tiles {i},{j} (check tile dimensions): {e}")
                     # Keep high diff

    best_arrangement = None
    min_score = float('inf')
    found_valid_arrangement = False
    for arrangement in permutations(range(n)):
        score = 0
        valid_arrangement = True
        try:
            for row in range(GRID_SIZE): # Horizontal
                for col in range(GRID_SIZE - 1):
                    left, right = arrangement[row*GRID_SIZE+col], arrangement[row*GRID_SIZE+col+1]
                    score += h_diff[left][right]
            for col in range(GRID_SIZE): # Vertical
                for row in range(GRID_SIZE - 1):
                    top, bottom = arrangement[row*GRID_SIZE+col], arrangement[(row+1)*GRID_SIZE+col]
                    score += v_diff[top][bottom]

            # Check if score is valid (not infinity)
            if np.isinf(score):
                 valid_arrangement = False
                 continue # Skip invalid arrangements caused by edge errors

            found_valid_arrangement = True
            if score < min_score:
                min_score = score
                best_arrangement = arrangement
        except IndexError:
             # Should not happen if permutations are correct, but good to have
             valid_arrangement = False
             continue

    if not found_valid_arrangement:
        print("Error: No valid arrangements found. Check edge difference calculations.")
        return None

    return best_arrangement # Returns (Pos -> Tile)

def assemble_preview_image(original_patches, arrangement):
    """Assembles PIL image (225x225) from 75x75 patches based on arrangement."""
    solved_puzzle_tiles = [None] * N_TILES
    for position_index, tile_index in enumerate(arrangement):
        if 0 <= tile_index < N_TILES:
            solved_puzzle_tiles[position_index] = original_patches[tile_index]
        else:
             solved_puzzle_tiles[position_index] = Image.new('RGB', (TILE_SIZE, TILE_SIZE), 'black')

    unused_tiles = [t for t in original_patches if t not in solved_puzzle_tiles]
    for i in range(N_TILES):
        if solved_puzzle_tiles[i] is None:
            if unused_tiles: solved_puzzle_tiles[i] = unused_tiles.pop(0)
            else: solved_puzzle_tiles[i] = Image.new('RGB', (TILE_SIZE, TILE_SIZE), 'grey')

    # Create final image by pasting tiles
    final_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
    for idx, tile in enumerate(solved_puzzle_tiles):
         row, col = divmod(idx, GRID_SIZE)
         paste_x, paste_y = col * TILE_SIZE, row * TILE_SIZE
         final_image.paste(tile, (paste_x, paste_y))
    return final_image

def scan_folder_for_images(folder_path):
    """Scan folder for image files and return list of image paths."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    
    # Filter out system files and thumbnails and remove duplicates
    filtered_files = []
    seen_files = set()
    
    for file_path in image_files:
        filename = os.path.basename(file_path)
        # Skip system files and thumbnails
        if (not filename.startswith(('.', 'Thumbs')) and 
            'thumbnail' not in filename.lower() and
            file_path not in seen_files):
            filtered_files.append(file_path)
            seen_files.add(file_path)
    
    return sorted(filtered_files)

def process_single_image(img_path, img_name):
    """Process a single image and return results."""
    try:
        scrambled_img_pil = Image.open(img_path)
        
        # Store original for display
        scrambled_img_display = scrambled_img_pil.copy()
        
        # Solve the puzzle
        patches = extract_tiles(scrambled_img_pil)
        if patches is None:
            raise ValueError("Failed to extract patches.")
            
        predicted_arrangement = solve_puzzle(patches)
        if predicted_arrangement is None:
            raise ValueError("Solver failed to find arrangement.")
            
        # Assemble solved image
        solved_img_pil = assemble_preview_image(patches, predicted_arrangement)
        
        # Create hackathon label
        hackathon_label = [-1] * N_TILES
        arrangement_list = list(predicted_arrangement)
        for original_patch_id in range(N_TILES):
            try: 
                final_position = arrangement_list.index(original_patch_id)
            except ValueError: 
                final_position = 0
            hackathon_label[original_patch_id] = final_position
            
        hackathon_label_str = ",".join(map(str, hackathon_label))
        
        return {
            'success': True,
            'scrambled_img': scrambled_img_display,
            'solved_img': solved_img_pil,
            'arrangement': predicted_arrangement,
            'label': hackathon_label_str,
            'image_name': img_name
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image_name': img_name
        }

def get_image_download_link(img, filename, text):
    """Generate a download link for an image."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# ---
# Streamlit App
# ---
def main():
    # Configure page
    st.set_page_config(
        page_title="Fragment Fusion",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme with neon green
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #00FF41;
        color: black;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00CC33;
        color: black;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    h1, h2, h3 {
        color: #00FF41;
    }
    .success-message {
        color: #00FF41;
        font-weight: bold;
    }
    .error-message {
        color: #FF6B6B;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧩 Fragment Fusion")
    st.markdown("### *Reconstructing Images from Fragments*")
    st.markdown("---")
    
    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = {}
    if 'image_files' not in st.session_state:
        st.session_state.image_files = []
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'show_images' not in st.session_state:
        st.session_state.show_images = True
    if 'show_csv' not in st.session_state:
        st.session_state.show_csv = False
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = None
    if 'num_to_solve' not in st.session_state:
        st.session_state.num_to_solve = 1
    if 'images_to_show' not in st.session_state:
        st.session_state.images_to_show = 5
    if 'results_to_show' not in st.session_state:
        st.session_state.results_to_show = 5
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Input Settings")
        
        # File uploader for cloud deployment
        st.subheader("Upload Images")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload all the fragmented images you want to reconstruct"
        )
        
        # Process uploaded files
        if uploaded_files:
            # Save uploaded files to a temporary directory
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Clear previous files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            
            # Save new files
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.session_state.selected_folder = temp_dir
            image_files = scan_folder_for_images(temp_dir)
            st.session_state.image_files = image_files
            st.session_state.num_to_solve = min(3, len(image_files))
            st.session_state.images_to_show = 5
            st.session_state.results_to_show = 5
            st.success(f"📁 {len(image_files)} images ready for processing")
        
        # Display folder info
        if st.session_state.selected_folder:
            st.info(f"Images loaded: {len(st.session_state.image_files)} files")
        
        # Number of images to solve
        if st.session_state.image_files:
            num_images = len(st.session_state.image_files)
            if num_images > 1:
                st.session_state.num_to_solve = st.number_input(
                    "Number of images to solve",
                    min_value=1,
                    max_value=num_images,
                    value=st.session_state.num_to_solve,
                    step=1,
                    help=f"Choose how many images to solve (1 to {num_images})"
                )
            else:
                # When only 1 image, just display the number
                st.session_state.num_to_solve = 1
                st.info(f"Solving 1 image")
        else:
            st.session_state.num_to_solve = 0
            
        # Output filename
        output_filename = st.text_input(
            "Output filename",
            value=DEFAULT_OUTPUT_FILENAME,
            help="Name for the CSV results file"
        )
        
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
    
    # Main content area
    if not st.session_state.image_files:
        st.info("👈 Please upload images using the file uploader in the sidebar to get started.")
        st.markdown("""
        ### How to use Fragment Fusion:
        1. Upload your fragmented images using the file uploader in the sidebar
        2. Select how many images you want to solve
        3. Click 'Start Fragment Fusion' to begin reconstructing fragments
        4. View results and download the reconstructed images and CSV file
        
        **Supported formats:** JPG, JPEG, PNG
        """)
        return
    
    # Show images with "Show More" functionality
    st.subheader("📷 Images Ready for Processing")
    
    # Calculate how many images to display
    total_images = len(st.session_state.image_files)
    images_to_display = min(st.session_state.images_to_show, total_images)
    
    # Display images in a grid
    cols = st.columns(4)
    for idx, img_path in enumerate(st.session_state.image_files[:images_to_display]):
        with cols[idx % 4]:
            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                st.image(img, caption=os.path.basename(img_path))
            except Exception as e:
                st.error(f"Error loading: {os.path.basename(img_path)}")
    
    # Show "Show More" button if there are more images to display
    if total_images > st.session_state.images_to_show:
        if st.button("🔄 Show More Images", use_container_width=True, key="show_more_images"):
            st.session_state.images_to_show += 5
            st.rerun()
    
    # Show message if all images are displayed
    if images_to_display == total_images and total_images > 5:
        st.success(f"✅ All {total_images} images are displayed")
    
    # Processing section
    st.subheader("⚙️ Processing")
    
    if st.button("🚀 Start Fragment Fusion", type="primary"):
        if not st.session_state.image_files:
            st.error("No images found to process!")
            return
            
        # Process selected number of images
        files_to_process = st.session_state.image_files[:st.session_state.num_to_solve]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, img_path in enumerate(files_to_process):
            img_name = os.path.basename(img_path)
            status_text.text(f"Processing {i+1}/{len(files_to_process)}: {img_name}")
            
            # Process image
            result = process_single_image(img_path, img_name)
            
            if result['success']:
                # Store in session state for display
                st.session_state.processed_results[img_name] = {
                    'scrambled': result['scrambled_img'],
                    'solved': result['solved_img'],
                    'arrangement': result['arrangement'],
                    'label': result['label']
                }
                results.append({
                    'image': img_name,
                    'label': result['label']
                })
                st.success(f"✅ {img_name} - Reconstructed!")
            else:
                st.error(f"❌ {img_name} - {result['error']}")
                # Add placeholder for failed images
                results.append({
                    'image': img_name,
                    'label': '0,0,0,0,0,0,0,0,0'
                })
            
            progress_bar.progress((i + 1) / len(files_to_process))
        
        status_text.text("Fragment Fusion complete!")
        
        # Create and store CSV
        if results:
            df = pd.DataFrame(results)
            st.session_state.csv_data = df
            csv = df.to_csv(index=False)
            
            # Download button for CSV
            st.download_button(
                label="📥 Download submission.csv",
                data=csv,
                file_name=output_filename,
                mime="text/csv",
                key="download_csv"
            )
    
    # Results display section
    if st.session_state.processed_results:
        st.subheader("🎯 Reconstruction Results")
        
        # Show processing summary
        st.info(f"Successfully reconstructed {len(st.session_state.processed_results)} images")
        
        # Option selection
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👁️ View Reconstructed Images", use_container_width=True):
                st.session_state.show_images = True
                st.session_state.show_csv = False
                st.rerun()
                
        with col2:
            if st.button("📊 View CSV File", use_container_width=True):
                st.session_state.show_images = False
                st.session_state.show_csv = True
                st.rerun()
        
        # Display based on selection
        if st.session_state.show_images:
            # Show solved images
            st.subheader("🖼️ Fragment Fusion Results")
            
            # Get list of processed image names
            processed_image_names = list(st.session_state.processed_results.keys())
            total_results = len(processed_image_names)
            results_to_display = min(st.session_state.results_to_show, total_results)
            
            # Display results with "Show More" functionality
            for img_name in processed_image_names[:results_to_display]:
                result = st.session_state.processed_results[img_name]
                st.markdown(f"**{img_name}**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Resize for display
                    display_scrambled = result['scrambled'].copy()
                    display_scrambled.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    st.image(display_scrambled, caption="Fragmented Image")
                
                with col2:
                    # Resize for display
                    display_solved = result['solved'].copy()
                    display_solved.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    st.image(display_solved, caption="Fused Reconstruction")
                    st.text(f"Fragment Arrangement: {result['arrangement']}")
                    
                    # Download link for solved image
                    download_link = get_image_download_link(
                        result['solved'], 
                        f"fused_{img_name}.png", 
                        "📥 Download Fused Image"
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Show "Show More" button for results if there are more to display
            if total_results > st.session_state.results_to_show:
                if st.button("🔄 Show More Results", use_container_width=True, key="show_more_results"):
                    st.session_state.results_to_show += 5
                    st.rerun()
            
            # Show message if all results are displayed
            if results_to_display == total_results and total_results > 5:
                st.success(f"✅ All {total_results} reconstruction results are displayed")
                
        elif st.session_state.show_csv:
            # Show CSV data
            st.subheader("📋 Fusion Results Data")
            if st.session_state.csv_data is not None:
                st.dataframe(st.session_state.csv_data, use_container_width=True)
                
                # Show raw CSV
                st.subheader("Raw CSV Data")
                st.code(st.session_state.csv_data.to_csv(index=False), language="csv")
                
                # Additional download button in CSV view
                csv_data = st.session_state.csv_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download submission.csv",
                    data=csv_data,
                    file_name=output_filename,
                    mime="text/csv",
                    key="download_csv_2"
                )

if __name__ == "__main__":
    main()