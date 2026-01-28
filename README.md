# ViPER
In the default_configs.yaml file, modify the datadir field to specify your dataset path and update the savedir field to set the directory for model checkpoints.
# Environments
pip install numpy pandas torch timm scikit-learn opencv-python huggingface_hub torchvision wandb matplotlib
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

# Run
python main.py --default_setting default_configs.yaml --dataname cifar10 --modelname vit_base_patch16_224 --prompt_type deep --prompt_tokens 10 --img_resize 224
