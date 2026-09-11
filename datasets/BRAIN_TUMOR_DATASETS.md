# Brain Tumor CT Scan Datasets

## **Important Medical Reality:**

### **X-RAY vs CT vs MRI for Brain Tumors:**

- **X-RAY**: ❌ **Cannot detect brain tumors** - X-rays cannot penetrate the skull bone
- **CT Scan**: ✅ **Can detect brain tumors** - Less detailed than MRI but useful
- **MRI**: ✅ **Best for brain tumors** - Most detailed imaging, gold standard

## **Brain Tumor CT Scan Datasets Found:**

### **1. IVY GAP (Glioblastoma MRI/CT)**
- **Source**: The Cancer Imaging Archive (TCIA)
- **Size**: 39 patients, 390 studies, 139.98GB
- **Type**: MRI + CT scans for glioblastoma patients
- **License**: NIH Controlled Data Access
- **Download**: https://www.cancerimagingarchive.net/collection/ivygap/
- **Instructions**: 
  1. Register on TCIA
  2. Apply for data access (controlled access)
  3. Download using TCIA Data Retriever
  4. Extract to: `datasets/ct_scan/ivy_gap/`

### **2. Brain Metastases Dataset (MRI/CT)**
- **Source**: Zenodo
- **Size**: 40 patients, 185 imaging studies, 838.5MB
- **Type**: MRI + CT scans for brain metastases
- **License**: Open access
- **Download**: https://zenodo.org/records/15499112
- **Instructions**:
  1. Download from Zenodo (free)
  2. Extract to: `datasets/ct_scan/brain_metastases/`

### **3. Germinoma Dataset (MRI/CT)**
- **Source**: Scientific Data (Nature)
- **Size**: 65 patients, 63 CT + 50 MRI datasets
- **Type**: MRI + CT scans for germinoma tumors
- **License**: Publicly available on figshare
- **Download**: Available via figshare repository
- **Instructions**:
  1. Access via figshare link from the paper
  2. Extract to: `datasets/ct_scan/germinoma/`

### **4. RFUds (Radiotherapy Follow-Up)**
- **Source**: Zenodo
- **Size**: 44 patients, 7.8GB
- **Type**: Pre-treatment CT + MRI for brain metastases
- **License**: Open access
- **Download**: https://zenodo.org/records/14525197
- **Instructions**:
  1. Download RFUds.zip from Zenodo
  2. Extract to: `datasets/ct_scan/rfuds/`

## **Recommended Dataset for Your Project:**

### **Best Choice: Brain Metastases Dataset (Zenodo)**
- **Why**: Free access, reasonable size (838.5MB), includes CT + MRI
- **Download**: https://zenodo.org/records/15499112
- **Contains detailed segmentations of brain metastases**

### **Alternative: RFUds Dataset**
- **Why**: Free access, smaller size (7.8GB), radiotherapy follow-up data
- **Download**: https://zenodo.org/records/14525197
- **Contains pre-treatment CT and MRI files**

## **Important Notes:**

1. **X-ray cannot detect brain tumors** - This is medically impossible
2. **CT scans can detect brain tumors** but are less detailed than MRI
3. **MRI remains the gold standard** for brain tumor detection
4. **Your current MRI system is already excellent** for brain tumor detection
5. **CT datasets are smaller** than lung cancer datasets because MRI is preferred

## **System Update Needed:**

The current CT model (TAP-CT) is trained for lung cancer. For brain tumor CT analysis, you would need:
- A brain tumor-specific CT model
- Training on brain tumor CT datasets
- Fine-tuning of existing models

## **Recommendation:**

**Keep your current MRI-based brain tumor detection** (it's excellent) and use CT datasets for:
- Brain metastases detection
- Radiotherapy planning
- Follow-up imaging
- Multi-modal analysis (MRI + CT combined)
