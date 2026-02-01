# 🛰️ Satellite Super-Resolution - Complete Colab Setup
# Run this notebook in Google Colab with GPU runtime

#@title 1️⃣ Clone Repository & Setup
#@markdown Run this cell first to clone your repo and install dependencies

# Clone your repository
!git clone https://github.com/Bharath-2005-07/ResolutionOf-Satellite.git
%cd ResolutionOf-Satellite

# Install dependencies
!pip install -q torch torchvision
!pip install -q opencv-python-headless pillow scikit-image
!pip install -q tqdm matplotlib
!pip install -q streamlit  # For local testing later

# Verify GPU
import torch
print(f"🖥️ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n✅ Setup complete!")
