"""
Download CT brain tumor dataset from Hugging Face
Dataset: Reddysridhar1/CT-Scan-Brain-Tumor
Size: 227 MB, 4,620 images
"""

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download
    print("Using huggingface_hub for download")
except ImportError:
    print("Installing huggingface_hub library...")
    import subprocess
    subprocess.check_call(["pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
import os

def download_ct_dataset():
    """Download CT brain tumor dataset from Hugging Face"""
    
    print("="*60)
    print("Downloading CT Brain Tumor Dataset")
    print("="*60)
    print("Source: Hugging Face - Reddysridhar1/CT-Scan-Brain-Tumor")
    print("Size: 227 MB")
    print("Images: 4,620 CT brain tumor images")
    print("="*60)
    
    # Create dataset directory
    dataset_dir = Path("datasets/ct_huggingface")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\nDownloading dataset from Hugging Face...")
        print("This may take a few minutes...")
        
        # Download using snapshot_download
        downloaded_path = snapshot_download(
            repo_id="Reddysridhar1/CT-Scan-Brain-Tumor",
            repo_type="dataset",
            local_dir=str(dataset_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Dataset downloaded successfully to {downloaded_path}")
        
        # List downloaded files
        print("\nAnalyzing downloaded files...")
        all_files = list(dataset_dir.rglob("*"))
        print(f"Total files downloaded: {len(all_files)}")
        
        # Count images
        image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"Image files: {len(image_files)}")
        
        if len(image_files) > 0:
            print(f"Sample image: {image_files[0].name}")
        
        print("\n" + "="*60)
        print("✓ Dataset download complete!")
        print("="*60)
        print(f"Dataset location: {dataset_dir}")
        print(f"Storage used: ~227 MB")
        print(f"Images available: {len(image_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_ct_dataset()
    if success:
        print("\n✅ Ready for training with new dataset!")
    else:
        print("\n❌ Dataset download failed")
