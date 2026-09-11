"""
Download sample medical imaging datasets for multi-modal detection
"""

import os
import requests
from pathlib import Path
import zipfile
import io

def download_sample_datasets():
    """Download sample medical imaging datasets"""
    
    base_dir = Path(__file__).parent
    ct_dir = base_dir / "ct_scan"
    xray_dir = base_dir / "x_ray"
    
    # Create subdirectories
    (ct_dir / "train").mkdir(parents=True, exist_ok=True)
    (ct_dir / "test").mkdir(parents=True, exist_ok=True)
    (xray_dir / "train").mkdir(parents=True, exist_ok=True)
    (xray_dir / "test").mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MEDICAL IMAGING DATASET DOWNLOAD")
    print("=" * 70)
    
    # Sample dataset URLs (small samples for testing)
    datasets = {
        "chest_xray_sample": {
            "url": "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/1-s2.0-S0929664620301806-main.pdf-1.jpg",
            "type": "xray",
            "description": "Sample chest X-ray image"
        },
        "ct_sample": {
            "url": "https://raw.githubusercontent.com/medpix/medpix/master/images/CT%20Brain%20Normal%20Axial%20slice.jpg",
            "type": "ct",
            "description": "Sample CT scan image"
        }
    }
    
    for name, info in datasets.items():
        print(f"\nDownloading {name}: {info['description']}")
        try:
            response = requests.get(info['url'], timeout=30)
            if response.status_code == 200:
                if info['type'] == 'xray':
                    save_path = xray_dir / "train" / f"{name}.jpg"
                else:
                    save_path = ct_dir / "train" / f"{name}.jpg"
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded to: {save_path}")
            else:
                print(f"✗ Failed to download {name}")
        except Exception as e:
            print(f"✗ Error downloading {name}: {str(e)}")
    
    print("\n" + "=" * 70)
    print("LARGE DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    
    print("\nFor large datasets, please download manually:")
    
    print("\n1. LIDC-IDRI (Lung CT Scans) - 133GB:")
    print("   URL: https://www.cancerimagingarchive.net/collection/lidc-idri/")
    print("   Instructions: Register on TCIA, use Data Retriever tool")
    print("   Extract to: datasets/ct_scan/lidc_idri/")
    
    print("\n2. Chest X-Ray Dataset (Taiwan) - Lung Cancer:")
    print("   URL: https://data.dmc.nycu.edu.tw/en/dataset/d11-x-ray")
    print("   Instructions: Register and download from portal")
    print("   Extract to: datasets/x_ray/taiwan_lung/")
    
    print("\n3. XCPData (Paired X-ray and CT) - 48 cases:")
    print("   URL: https://drive.google.com/file/d/1RNiZL4dt7BZv_QRc650wNZv8jviNvhHw/view")
    print("   Instructions: Download from Google Drive")
    print("   Extract to: datasets/xcp_data/")
    
    print("\n4. AISD (Stroke CT) - 397 scans:")
    print("   URL: https://drive.google.com/file/d/157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk/view")
    print("   Instructions: Download image.zip and mask.zip")
    print("   Extract to: datasets/ct_scan/aisd/")
    
    print("\n" + "=" * 70)
    print("DATASET STRUCTURE CREATED")
    print("=" * 70)
    print(f"CT Scan Directory: {ct_dir}")
    print(f"X-Ray Directory: {xray_dir}")

if __name__ == '__main__':
    download_sample_datasets()
