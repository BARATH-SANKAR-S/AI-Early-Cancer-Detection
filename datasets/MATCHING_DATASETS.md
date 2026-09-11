# Brain Tumor Datasets - Matching Your Current MRI Dataset Size

## **Your Current MRI Dataset:**
- **Size**: ~1600 images
- **Types**: Glioma, Meningioma, Pituitary, No Tumor
- **Modality**: MRI scans

## **Matching CT Scan Dataset:**

### **1. Ultralytics Brain Tumor Dataset (RECOMMENDED)**
- **Size**: 1,116 images (closest to your 1600)
- **Types**: MRI + CT scans
- **Classes**: Negative (no tumor), Positive (tumor present)
- **Split**: 893 training, 223 validation
- **Size**: 4.21 MB (very manageable)
- **Download**: https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip
- **License**: Open source
- **Instructions**:
  1. Download from GitHub link above
  2. Extract to: `datasets/ct_scan/ultralytics_brain_tumor/`
  3. Contains both MRI and CT images for brain tumor detection

### **2. Research CT Dataset (900 images)**
- **Size**: 900 CT scan images (150 per 6 tumor types)
- **Types**: 6 brain tumor types (Meningioma, Schwannoma, Neurofibromatosis, Glioma, Chondrosarcoma, Chordoma)
- **Resolution**: 512×512 pixels
- **Total attributes**: 3,600 (900 images × 4 ROIs)
- **Source**: Radiopaedia repository
- **Status**: Research dataset (may require access request)
- **Paper**: https://doi.org/10.1038/s41598-025-16825-5

## **X-Ray Brain Tumor Dataset:**

### **❌ MEDICAL REALITY: X-RAY CANNOT DETECT BRAIN TUMORS**

**Why no X-ray brain tumor datasets exist:**
- X-rays cannot penetrate the human skull bone
- Skull blocks X-rays, making brain imaging impossible
- MRI and CT are the only modalities for brain tumor detection
- This is a fundamental limitation of X-ray technology

**What X-ray is used for:**
- Lung cancer detection
- Bone fractures
- Chest imaging
- Dental imaging

**What you have instead:**
- Your MRI dataset (excellent for brain tumors)
- CT scan dataset (good for brain tumors)
- X-ray dataset (for lung cancer only)

## **Download Instructions for CT Dataset:**

### **Ultralytics Brain Tumor Dataset (Recommended):**

```bash
# Download directly
cd datasets/ct_scan
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip
unzip brain-tumor.zip
```

**Or manually:**
1. Go to: https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip
2. Download the ZIP file
3. Extract to: `datasets/ct_scan/ultralytics_brain_tumor/`

## **Dataset Comparison:**

| Dataset | Modality | Images | Classes | Size | Download |
|---------|----------|--------|---------|------|----------|
| Your Current | MRI | ~1600 | 4 classes | Medium | Already have |
| Ultralytics | MRI+CT | 1,116 | 2 classes | 4.21 MB | Free |
| Research CT | CT | 900 | 6 classes | Medium | Research access |
| Brain Metastases | MRI+CT | 185 studies | Multiple | 838.5 MB | Free |

## **Recommendation:**

**Download the Ultralytics Brain Tumor Dataset** because:
- ✅ Closest size to your current dataset (1,116 vs 1,600)
- ✅ Free and easy to download (4.21 MB)
- ✅ Contains both MRI and CT images
- ✅ Ready-to-use format
- ✅ Open source license

## **Important Note:**

You will **not find X-ray brain tumor datasets** because they medically cannot exist. X-rays cannot see through the skull, so brain tumor detection requires MRI or CT scans only.

Your system is correctly configured:
- **MRI**: Brain tumor detection (your current dataset)
- **CT**: Brain tumor detection (Ultralytics dataset)
- **X-ray**: Lung cancer detection (separate functionality)
