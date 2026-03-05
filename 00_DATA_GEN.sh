#!/bin/bash

# ================= setting zone =================

# python script path
PYTHON_SCRIPT="/workspaces/data/XCM_preprocess/10_gen_various_SR.py"


# INPUT_PARQUET="/workspaces/data/pure_bird_manifests/pure_class_A_Global_long_10s.parquet"
INPUT_PARQUET="/workspaces/data/pure_bird_manifests/pure_class_A_dense_seeds.parquet"
# output path
BASE_DIR_NAME="/workspaces/bird_data/data_gen"

# the overlap ratios to generate, you can modify this list as needed
RATIOS=(0.0 0.1 0.2) # 您可以根据需要在这里添加更多比例，例如 (0.0 0.2 0.5)

# ===========================================

# check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: 找不到文件 $PYTHON_SCRIPT"
    exit 1
fi

echo "========================================================"
echo "   The ratios to generate: ${RATIOS[*]}"
echo "========================================================"


for ratio in "${RATIOS[@]}"; do

    suffix=$(awk -v r="$ratio" 'BEGIN {printf "%02.0f", r*100}')


    CURRENT_OUTPUT_DIR="${BASE_DIR_NAME}_OVERLAP_${suffix}"

    echo ""
    echo "--------------------------------------------------------"
    echo "▶️  Processing Overlap Ratio: $ratio (Suffix: $suffix)"
    echo "📂 Target Path: $CURRENT_OUTPUT_DIR"
    echo "--------------------------------------------------------"

    # Run the Python script with the current ratio and output directory

    python "$PYTHON_SCRIPT" \
        --output_root "$CURRENT_OUTPUT_DIR" \
        --overlap_ratio "$ratio" \
        --input_parquet "$INPUT_PARQUET"

    # Check if the previous step was successful
    if [ $? -eq 0 ]; then
        echo "✅ Success: Ratio $ratio -> Folder suffix $suffix"
    else
        echo "❌ Error occurred: Ratio $ratio"
        exit 1
    fi

    # Buffer time before the next iteration
    sleep 2
done

echo ""
echo "========================================================"
echo "🎉 All tasks completed!"
echo "========================================================"