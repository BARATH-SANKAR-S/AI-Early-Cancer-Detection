# Module 5: Publicly Available Cancer Detection Datasets

## Why No Erode-Specific Dataset Exists

There is no publicly available cancer detection dataset specifically for Erode District, Tamil Nadu, India, for the following reasons:

1. **Privacy and Data Protection Regulations**: Medical data is subject to strict privacy laws (HIPAA in US, GDPR in Europe, similar regulations in India). Sharing patient data requires extensive de-identification and consent processes.

2. **Limited Digital Infrastructure**: Erode District, like many regions in India, may not have comprehensive digital medical record systems that would enable large-scale data collection and sharing.

3. **Regional Data Collection Challenges**: Most publicly available medical imaging datasets are collected from major research institutions, university hospitals, or national screening programs in developed countries (US, Europe, etc.).

4. **Resource Constraints**: Creating and curating a high-quality medical dataset requires significant funding, expertise, and infrastructure that may not be available at the district level.

5. **Ethical and Legal Barriers**: Sharing medical data across borders requires complex ethical approvals and legal agreements that are difficult to navigate.

## Recommended Alternative Datasets

Since no Erode-specific dataset exists, we will use internationally recognized datasets that are suitable for training AI models. These datasets can be adapted for the Erode context through transfer learning and fine-tuning.

---

## Dataset 1: LIDC-IDRI (Lung Cancer)

### Dataset Name
LIDC-IDRI (Lung Image Database Consortium - Image Database Resource Initiative)

### Source
The Cancer Imaging Archive (TCIA)
National Cancer Institute (NCI)

### Download Link
https://www.cancerimagingarchive.net/collection/lidc-idri/

### Number of Records
- **Subjects**: 1,010 patients
- **Studies**: 1,308 CT scans
- **Series**: 1,308
- **Images**: 244,527 CT images
- **Size**: 133.16 GB

### Features
- **Image Type**: Computed Tomography (CT) scans
- **Annotations**: XML files with radiologist annotations
- **Nodule Categories**: 
  - Nodule ≥ 3mm
  - Nodule < 3mm
  - Non-nodule ≥ 3mm
- **Radiologist Reviews**: 4 experienced thoracic radiologists annotated each scan
- **Two-phase annotation**: Blinded read followed by unblinded read
- **Metadata**: Patient demographics, scan parameters

### Target Variable
- **Primary**: Lung nodule classification (benign vs malignant)
- **Secondary**: Nodule size, location, characteristics

### License
CC BY 3.0 (Creative Commons Attribution 3.0)

### Why It Is Suitable
1. **Large Scale**: Over 1,000 patients provide sufficient training data
2. **High Quality**: Annotated by 4 expert radiologists with consensus process
3. **Standard Format**: DICOM format, widely used in medical imaging
4. **Public Access**: Free to download with permissive license
5. **Comprehensive**: Includes both images and detailed annotations
6. **Research-Backed**: Developed by NCI with academic and industry collaboration
7. **Relevant**: Specifically designed for lung cancer detection research

---

## Dataset 2: CBIS-DDSM (Breast Cancer)

### Dataset Name
CBIS-DDSM (Curated Breast Imaging Subset of Digital Database for Screening Mammography)

### Source
The Cancer Imaging Archive (TCIA)

### Download Link
https://www.cancerimagingarchive.net/collection/cbis-ddsm/

### Number of Records
- **Subjects**: 1,566 patients
- **Images**: 2,620 mammography studies
- **Size**: 163.51 GB

### Features
- **Image Type**: Digital mammography (converted to DICOM)
- **Categories**: 
  - Normal cases
  - Benign cases
  - Malignant cases
- **Annotations**: 
  - ROI (Region of Interest) segmentation
  - Bounding boxes
  - Pathology information
- **Views**: Standard mammography views (CC, MLO)
- **Quality**: Curated and standardized by trained mammographer

### Target Variable
- **Primary**: Breast cancer classification (benign vs malignant)
- **Secondary**: Lesion type, location, BI-RADS assessment

### License
CC BY 3.0 (Creative Commons Attribution 3.0)

### Why It Is Suitable
1. **Curated Quality**: Expert-curated subset of DDSM with improved annotations
2. **Standard Format**: DICOM format for easy integration
3. **Ground Truth**: Verified pathology information
4. **Balanced**: Includes normal, benign, and malignant cases
5. **Research-Ready**: Designed specifically for CAD/CADe system development
6. **Public Access**: Free download with permissive license
7. **Comprehensive**: Includes full mammograms and ROI crops

---

## Dataset 3: ISIC-DICM-17K (Skin Cancer)

### Dataset Name
ISIC-DICM-17K (ISIC Dermoscopic Images and Clinical Metadata 17K)

### Source
International Skin Imaging Collaboration (ISIC) Archive

### Download Link
https://api.isic-archive.com/doi/isic-dicm-17k/

### Number of Records
- **Images**: 17,060 dermoscopic images
- **Classes**: 8,530 melanoma, 8,530 non-melanoma (balanced)
- **Size**: 6.9 GB (complete bundle)

### Features
- **Image Type**: Dermoscopic images
- **Clinical Metadata**: Age, sex, lesion location, diagnosis
- **Image Quality**: High-resolution dermoscopic photography
- **Balanced Dataset**: Equal numbers of melanoma and non-melanoma cases
- **Metadata**: CSV file with clinical information

### Target Variable
- **Primary**: Melanoma vs non-melanoma classification
- **Secondary**: Lesion characteristics, clinical metadata

### License
CC BY 3.0 (Creative Commons Attribution 3.0)

### Why It Is Suitable
1. **Balanced**: Equal distribution of classes prevents bias
2. **High Quality**: Professional dermoscopic imaging
3. **Clinical Metadata**: Includes relevant patient information
4. **Moderate Size**: 17K images is manageable for training
5. **Research-Backed**: Curated from ISIC Archive by researchers
6. **Public Access**: Free download with permissive license
7. **Well-Documented**: GitHub repository with usage examples

---

## Dataset 4: Duke Lung Cancer Screening Dataset 2024

### Dataset Name
Duke Lung Cancer Screening Dataset 2024 (DLCS-2024)

### Source
Duke University Health System
Published on Zenodo

### Download Link
Part 1: https://zenodo.org/records/13799069
Part 2: https://zenodo.org/records/12784601
Part 3: https://zenodo.org/records/14659131

### Number of Records
- **CT Volumes**: 1,613 CT volumes
- **Annotated Nodules**: 2,487 nodules
- **Patients**: 2,061 patients (subset used for dataset)
- **Size**: Multi-part dataset (large)

### Features
- **Image Type**: Low-dose CT screening scans
- **Annotations**: 3D bounding boxes for lung nodules
- **Semi-Automated**: Deep learning detection with radiologist verification
- **Accuracy**: >90% annotation accuracy confirmed by spot-checking
- **Current Technology**: Reflects modern CT imaging (2015-2021)
- **Metadata**: CT image info, patient demographics

### Target Variable
- **Primary**: Lung nodule classification and risk assessment
- **Secondary**: Nodule location, size, characteristics

### License
CC BY 4.0 (Creative Commons Attribution 4.0)

### Why It Is Suitable
1. **Modern Data**: Recent scans (2015-2021) with current CT technology
2. **Large Scale**: Over 1,600 CT volumes with 2,400+ annotated nodules
3. **High Quality**: Semi-automated annotation with >90% accuracy
4. **Screening Context**: Specifically designed for lung cancer screening
5. **Current Relevance**: Reflects modern clinical practice
6. **Public Access**: Available on Zenodo with permissive license
7. **Well-Documented**: Comprehensive metadata and documentation

---

## Summary of Recommended Datasets

| Dataset | Cancer Type | Records | Size | License | Priority |
|---------|-------------|---------|------|---------|----------|
| LIDC-IDRI | Lung | 1,010 patients, 244K images | 133 GB | CC BY 3.0 | High |
| CBIS-DDSM | Breast | 1,566 patients, 2,620 studies | 164 GB | CC BY 3.0 | High |
| ISIC-DICM-17K | Skin | 17,060 images | 6.9 GB | CC BY 3.0 | Medium |
| Duke DLCS-2024 | Lung | 1,613 CT volumes, 2,487 nodules | Large | CC BY 4.0 | High |

**Recommended Implementation Strategy:**
1. Start with ISIC-DICM-17K (skin cancer) - smaller size, easier to handle
2. Implement LIDC-IDRI (lung cancer) - comprehensive lung cancer dataset
3. Add CBIS-DDSM (breast cancer) - for mammography-based detection
4. Consider Duke DLCS-2024 for modern lung cancer screening models

All datasets use permissive Creative Commons licenses, making them suitable for research and educational purposes.
