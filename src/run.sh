#!/bin/bash
# run.sh: A three-step image processing pipeline.
# Step 1: Align all images from the input directory and save to a new output directory.
# Step 2: Generate an HDR image from the aligned images.
# Step 3: Apply tone mapping to the HDR image.
#
# Usage:
#   ./run.sh
#
# Make sure the following Python scripts are in the same directory:
#   - alignment.py
#   - generate_hdr.py
#   - tone_mapping.py
#
# Also, ensure your input images are in the "input_images" directory.

# Alignment ---
INPUT_DIR="input_images"
ALIGNED_DIR="aligned_images"

# Create the aligned images directory if it doesn't exist.
mkdir -p "$ALIGNED_DIR"
echo "Step 1: Running alignment on images in '$INPUT_DIR'..."
python3 alignment_MTB.py --input_dir "$INPUT_DIR" --output_dir "$ALIGNED_DIR"
if [ $? -ne 0 ]; then
    echo "Alignment step failed." >&2
    exit 1
fi
echo "Alignment completed. Aligned images saved in '$ALIGNED_DIR'."

# HDR Generation ---
HDR_OUTPUT="output.hdr"
echo "Step 2: Generating HDR image from aligned images in '$ALIGNED_DIR'..."
python3 Robertson.py --input_dir "$ALIGNED_DIR" --output_hdr "$HDR_OUTPUT"
if [ $? -ne 0 ]; then
    echo "HDR generation step failed." >&2
    exit 1
fi
echo "HDR generation completed. HDR image saved as '$HDR_OUTPUT'."

# Tone Mapping ---
LDR_OUTPUT="output_tonemapped.jpg"
echo "Step 3: Applying tone mapping on HDR image '$HDR_OUTPUT'..."
python3 tone_mapping.py --input_hdr "$HDR_OUTPUT" --output "$LDR_OUTPUT"
if [ $? -ne 0 ]; then
    echo "Tone mapping step failed." >&2
    exit 1
fi
echo "Tone mapping completed. Tone-mapped image saved as '$LDR_OUTPUT'."
