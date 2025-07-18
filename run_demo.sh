export FLORENCE2_MODEL_PATH="./checkpoints/Florence-2-large"
export SAM2_MODEL_PATH="./checkpoints/sam2.1_hiera_large.pt"
export FACE_ID_MODEL_PATH="./checkpoints/model_ir_se50.pth"
export CLIP_MODEL_PATH="./checkpoints/clip-vit-large-patch14"
export FLUX_MODEL_PATH="./checkpoints/FLUX.1-dev"
# export FLUX_MODEL_PATH="./checkpoints/FLUX.1-dev-bnb-4bit"
# export FLUX_TRANSFORMERS_PATH="./checkpoints/FLUX.1-dev-gguf/flux1-dev-Q3_K_S.gguf"
# export FLUX_MODEL_PATH="./checkpoints/FLUX.1-schnell"
export DPG_VQA_MODEL_PATH="./checkpoints/mplug_visual-question-answering_coco_large_en"
export DINO_MODEL_PATH="./checkpoints/dino-vits16"

python run_gradio.py