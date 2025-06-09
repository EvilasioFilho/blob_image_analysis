"""
Particle Detection and Aggregation Analysis
-------------------------------------------
This script processes microscopy images to detect particles/blobs and analyze their
aggregation patterns. It uses computer vision techniques to identify individual
particles and determine their relationships.

Author: Evilasio Anisio Costa Filho
      │ e: evilasio.costa@certbio.ufcg.edu.br
      │ a: Rua Aprígio Veloso, 882 - Bloco CN - Térreo - Universitário
      │    Campina Grande - PB, Brasil
      │ w: certbio.net
      │ ORCID: 0000-0002-2401-0663
      │ Lattes: http://lattes.cnpq.br/8131766993264892
Date: June 2025

Usage:
    python blob_detector_with_aggregation.py --image images/om/250X1.png --output results.xlsx
    python blob_detector_with_aggregation.py --help
"""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label
from scipy.spatial.distance import cdist  # For nearest neighbor distance
import pandas as pd
from collections import deque  # For BFS in aggregation analysis
import argparse  # For command-line arguments
import os  # For file path operations

# -----------------------------------------------------------------------------
# CONSTANTS AND CONFIGURATION
# -----------------------------------------------------------------------------

# Conversion factors between pixels and micrometers
PX_PER_UM = 100 / 130  # pixels per micrometer (calibration: 100px = 130µm)
UM_PER_PX = 1 / PX_PER_UM  # micrometers per pixel

# Aggregation analysis parameters
AGGREGATION_CONTACT_FACTOR = 1.1  # Factor for defining "touching" particles

# Color palette for visualization (BGR format in OpenCV)
COLOR_PALETTE = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Dark Blue
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Red
    (128, 128, 0),  # Teal
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Olive
]
ISOLATED_PARTICLE_COLOR = (200, 200, 200)  # Light Gray for isolated particles
TEXT_COLOR = (0, 0, 0)  # Black for text labels

# -----------------------------------------------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------------------------------------------

def preprocess_image(image_path, padding=5):
    """
    Load and preprocess the microscopy image for blob detection.

    Args:
        image_path: Path to the input image
        padding: Number of pixels to pad around the image

    Returns:
        Tuple of (original image, padded image, binary mask, distance transform)
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at '{image_path}'.")

    # Add padding to help with edge detection
    img_padded = cv2.copyMakeBorder(img, padding, padding, padding, padding,
                                    cv2.BORDER_REFLECT)

    # Convert to HSV and threshold the value channel
    hsv = cv2.cvtColor(img_padded, cv2.COLOR_BGR2HSV)
    _, binary = cv2.threshold(hsv[:, :, 2], 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # Create distance transform for better particle separation
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    return img, img_padded, morph, dist, dist_norm

# -----------------------------------------------------------------------------
# PARTICLE DETECTION
# -----------------------------------------------------------------------------

def detect_particles(padded_image, morph_mask, dist_transform, dist_norm, padding=5):
    """
    Detect individual particles in the image using distance transform and
    contour analysis.

    Args:
        padded_image: The padded input image
        morph_mask: Binary mask after morphological operations
        dist_transform: Distance transform of the binary mask
        dist_norm: Normalized distance transform
        padding: Padding size used in preprocessing

    Returns:
        List of detected particles with their properties
    """
    # Find local maxima in the distance transform
    max_filt_size = 17  # Size of the maximum filter window
    max_filt = maximum_filter(dist_norm, size=max_filt_size)
    local_max_threshold = 0.1  # Minimum threshold for local maxima
    local_max = (dist_norm == max_filt) & (dist_norm > local_max_threshold)
    labeled_maxima, num_potential_centers = label(local_max)

    # Get original image dimensions
    original_h, original_w = padded_image.shape[:2]
    original_h -= 2 * padding
    original_w -= 2 * padding

    # Process each potential particle center
    results = []
    for i in range(1, num_potential_centers + 1):
        # Find the coordinates of the current local maximum
        mask = (labeled_maxima == i)
        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            continue

        # Find the point with maximum distance value
        max_dist_in_region = 0
        center_y, center_x = -1, -1
        for r_coord, c_coord in coords:
            if dist_transform[r_coord, c_coord] > max_dist_in_region:
                max_dist_in_region = dist_transform[r_coord, c_coord]
                center_y, center_x = r_coord, c_coord

        if center_y == -1:
            continue

        # Estimated radius from the distance transform
        r_est_px = int(max_dist_in_region)

        # Only keep particles whose centers are inside the original image
        if r_est_px > 1 and padding <= center_x < original_w + padding and padding <= center_y < original_h + padding:
            # Create a circular mask and refine it with the morphological mask
            blob_mask = np.zeros_like(morph_mask, dtype=np.uint8)
            cv2.circle(blob_mask, (center_x, center_y), r_est_px, 255, -1)
            blob_mask_refined = cv2.bitwise_and(morph_mask, blob_mask)

            # Find contours in the refined mask
            cnts, _ = cv2.findContours(blob_mask_refined, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

            if cnts:
                # Take the largest contour
                cnt = max(cnts, key=cv2.contourArea)
                area_px = cv2.contourArea(cnt)

                # Filter out very small particles (likely noise)
                if area_px < 5:
                    continue

                # Calculate particle properties
                perimeter = cv2.arcLength(cnt, True)
                diameter_px = np.sqrt(4 * area_px / np.pi)
                radius_px = diameter_px / 2
                circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
                area_um2 = area_px * (UM_PER_PX ** 2)
                diameter_um = diameter_px * UM_PER_PX

                # Find the center of mass using moments
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx_contour_padded = int(M["m10"] / M["m00"])
                    cy_contour_padded = int(M["m01"] / M["m00"])
                else:
                    cx_contour_padded, cy_contour_padded = center_x, center_y

                # Store the particle data
                results.append({
                    'id': len(results),
                    'x_px_orig': cx_contour_padded - padding,  # Original image coord
                    'y_px_orig': cy_contour_padded - padding,  # Original image coord
                    'x_px_pad': cx_contour_padded,  # Padded image coord
                    'y_px_pad': cy_contour_padded,  # Padded image coord
                    'area_px': area_px,
                    'area_um2': area_um2,
                    'diameter_px': diameter_px,
                    'diameter_um': diameter_um,
                    'radius_px': radius_px,
                    'circularity': circularity * 100  # As percentage
                })

    return results

# -----------------------------------------------------------------------------
# AGGREGATION ANALYSIS
# -----------------------------------------------------------------------------

def analyze_aggregation(particles_df, coordinates):
    """
    Analyze the aggregation patterns of particles.

    Args:
        particles_df: DataFrame containing particle information
        coordinates: Array of particle coordinates

    Returns:
        Updated DataFrame with aggregation information and summary statistics
    """
    num_particles = len(particles_df)

    # Create adjacency list for particles that are close enough to be "touching"
    adj = [[] for _ in range(num_particles)]
    particle_radii_px = particles_df['radius_px'].values

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Calculate distance between particle centers
            p1_coord = coordinates[i]
            p2_coord = coordinates[j]
            dist_centers = np.linalg.norm(p1_coord - p2_coord)

            # Check if particles are close enough to be considered "touching"
            if dist_centers < (particle_radii_px[i] + particle_radii_px[j]) * AGGREGATION_CONTACT_FACTOR:
                adj[i].append(j)
                adj[j].append(i)

    # Identify connected components (aggregates) using BFS
    visited = [False] * num_particles
    aggregate_id_counter = 0
    particles_df['aggregate_id'] = -1
    particles_df['particles_in_aggregate'] = 0

    aggregates_summary = []

    for i in range(num_particles):
        if not visited[i]:
            current_aggregate_particles = []
            q = deque()
            q.append(i)
            visited[i] = True
            aggregate_id_counter += 1  # Starts at 1 for the first aggregate

            # BFS to find all particles in this aggregate
            while q:
                u = q.popleft()
                current_aggregate_particles.append(u)
                particles_df.loc[u, 'aggregate_id'] = aggregate_id_counter

                for v_neighbor in adj[u]:
                    if not visited[v_neighbor]:
                        visited[v_neighbor] = True
                        q.append(v_neighbor)

            # Update number of particles in this aggregate
            num_in_agg = len(current_aggregate_particles)
            for particle_idx in current_aggregate_particles:
                particles_df.loc[particle_idx, 'particles_in_aggregate'] = num_in_agg

            # Only count as an aggregate if it has more than one particle
            if num_in_agg > 1:
                aggregates_summary.append({
                    'agg_id': aggregate_id_counter,
                    'num_particles': num_in_agg
                })

    return particles_df, aggregates_summary

# -----------------------------------------------------------------------------
# NEAREST NEIGHBOR DISTANCE
# -----------------------------------------------------------------------------

def calculate_nnd(particles_df, coordinates):
    """
    Calculate the Nearest Neighbor Distance (NND) for each particle.

    Args:
        particles_df: DataFrame containing particle information
        coordinates: Array of particle coordinates

    Returns:
        DataFrame updated with NND information
    """
    num_particles = len(particles_df)

    if num_particles > 1:
        # Calculate pairwise distances between all particles
        dist_matrix = cdist(coordinates, coordinates)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances

        # Find the minimum distance for each particle
        nnd_px = np.min(dist_matrix, axis=1)
        particles_df['nnd_px'] = nnd_px
        particles_df['nnd_um'] = nnd_px * UM_PER_PX
    else:
        # Handle case with only one or zero particles
        particles_df['nnd_px'] = np.nan if num_particles == 1 else pd.NA
        particles_df['nnd_um'] = np.nan if num_particles == 1 else pd.NA

    return particles_df

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

def visualize_particles(original_image, padded_image, particles_df, padding=5):
    """
    Create a visualization of the detected particles, colored by aggregate.

    Args:
        original_image: The original input image
        padded_image: The padded version of the image
        particles_df: DataFrame containing particle information
        padding: Padding size used in preprocessing

    Returns:
        Image with visualized particles
    """
    # Create a copy of the original image for drawing
    original_h, original_w = original_image.shape[:2]
    result_image = padded_image[padding:padding + original_h,
                              padding:padding + original_w].copy()

    # Draw each particle
    for _, row in particles_df.iterrows():
        # Get particle properties
        center_x = int(row['x_px_orig'])
        center_y = int(row['y_px_orig'])
        radius = int(row['radius_px'])
        num_in_agg = row['particles_in_aggregate']
        agg_id = row['aggregate_id']

        # Choose color based on aggregate status
        if num_in_agg > 1:
            # Use color from palette for aggregated particles
            color_index = int((agg_id - 1) % len(COLOR_PALETTE))
            draw_color = COLOR_PALETTE[color_index]
        else:
            # Use gray for isolated particles
            draw_color = ISOLATED_PARTICLE_COLOR

        # Draw circle representing the particle
        cv2.circle(result_image, (center_x, center_y), radius, draw_color, 2)

        # Uncomment to add particle ID labels
        # cv2.putText(result_image, str(int(row['id'])),
        #            (center_x - 7, center_y - 7),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_COLOR, 1)

    return result_image

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

def main():
    """Main function to run the particle analysis pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Particle Detection and Aggregation Analysis')
    parser.add_argument('--image', type=str, default='images/om/250X1.png',
                        help='Path to the input image (default: images/om/250X1.png)')
    parser.add_argument('--output', type=str, default='blob_results_extended_colored.xlsx',
                        help='Path to save the output Excel file (default: blob_results_extended_colored.xlsx)')
    parser.add_argument('--padding', type=int, default=5,
                        help='Padding size in pixels (default: 5)')
    parser.add_argument('--contact-factor', type=float, default=1.1,
                        help='Aggregation contact factor - multiplier for determining particle contact (default: 1.1)')
    parser.add_argument('--save-image', type=str, default='',
                        help='Save the visualization image to the specified path (optional)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display the visualization window (useful for batch processing)')

    args = parser.parse_args()

    # Configuration from arguments
    image_path = args.image
    output_excel = args.output
    padding = args.padding

    # Update global constant if provided
    global AGGREGATION_CONTACT_FACTOR
    if args.contact_factor != 1.1:  # If user specified a different value
        AGGREGATION_CONTACT_FACTOR = args.contact_factor
        print(f"Using custom aggregation contact factor: {AGGREGATION_CONTACT_FACTOR}")

    # Validate file paths
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at '{image_path}'.")
        return

    output_dir = os.path.dirname(output_excel)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Processing image: {image_path}")
    print(f"Results will be saved to: {output_excel}")

    # Step 1: Preprocess the image
    try:
        print("Preprocessing image...")
        original_image, padded_image, morph_mask, dist_transform, dist_norm = preprocess_image(image_path, padding)
        original_h, original_w = original_image.shape[:2]
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return

    # Step 2: Detect particles
    print("Detecting particles...")
    particles = detect_particles(padded_image, morph_mask, dist_transform, dist_norm, padding)
    particles_df = pd.DataFrame(particles)
    num_particles = len(particles_df)

    if num_particles == 0:
        print("No particles detected.")
        return

    # Step 3: Calculate image statistics
    original_image_area_px = original_h * original_w
    original_image_area_um2 = original_image_area_px * (UM_PER_PX ** 2)
    particle_density_per_um2 = num_particles / original_image_area_um2

    print(f"Particle Density: {particle_density_per_um2:.4e} particles/µm²")
    print(f"Total particles found: {num_particles}")

    # Step 4: Calculate nearest neighbor distances
    if num_particles > 1:
        coordinates = particles_df[['x_px_orig', 'y_px_orig']].values
        particles_df = calculate_nnd(particles_df, coordinates)

    # Step 5: Analyze aggregation patterns
    if num_particles > 1:
        particles_df, aggregates = analyze_aggregation(particles_df, coordinates)

        # Calculate aggregation statistics
        num_aggregates = len(aggregates)
        particles_in_aggregates = particles_df[particles_df['particles_in_aggregate'] > 1]['id'].count()
        isolated_particles = num_particles - particles_in_aggregates

        print(f"\nAggregation Analysis (Factor: {AGGREGATION_CONTACT_FACTOR}):")
        print(f"  Number of aggregates (>=2 particles): {num_aggregates}")
        print(f"  Total particles involved in aggregates: {particles_in_aggregates}")
        print(f"  Number of isolated particles: {isolated_particles}")

        if num_aggregates > 0:
            avg_particles_per_aggregate = particles_in_aggregates / num_aggregates
            print(f"  Average particles per aggregate: {avg_particles_per_aggregate:.2f}")
    else:
        particles_df['aggregate_id'] = 1 if num_particles == 1 else pd.NA
        particles_df['particles_in_aggregate'] = 1 if num_particles == 1 else 0
        print("\nAggregation Analysis: Not applicable (less than 2 particles).")

    # Step 6: Visualize results
    result_image = visualize_particles(original_image, padded_image, particles_df, padding)

    # Step 7: Save results to Excel
    df_excel = particles_df.copy()
    df_excel.to_excel(output_excel, index=False)
    print(f"\nResults saved to {output_excel}")

    # Optionally save the visualization image
    if args.save_image:
        try:
            cv2.imwrite(args.save_image, result_image)
            print(f"Visualization saved to {args.save_image}")
        except Exception as e:
            print(f"Error saving visualization image: {str(e)}")

    # Step 8: Display the results (unless --no-display is specified)
    if not args.no_display:
        cv2.imshow('Aggregates Colored', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Skipping visualization display (--no-display specified)")

# Run the script
if __name__ == "__main__":
    main()
