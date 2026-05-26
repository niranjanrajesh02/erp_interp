CONFIG=2
SPLIT="train"

# pixel, low, mid-level feature banks 
# uv run python feature_banks/create_plm_banks.py \
#     --stim-metadata-path data/erp_processed/config${CONFIG}/sub-01_${SPLIT}_stim_metadata.h5 \
#     --stim-dir data/THINGS_images/images/object_images \
#     --out-dir feature_banks \
#     --max-feature-dim 100 \
#     --device-id 1

# dnn feature banks
uv run python feature_banks/create_dnn_banks.py \
    --stim-metadata-path data/erp_processed/config${CONFIG}/sub-01_${SPLIT}_stim_metadata.h5 \
    --stim-dir data/THINGS_images/images/object_images \
    --out-dir feature_banks/dnn_banks/ \
    --max-feature-dim 100 \
    --device-id 1 \


