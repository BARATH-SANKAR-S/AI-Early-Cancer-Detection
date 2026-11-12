"""
Clinical Oncology Knowledge Base & Semantic Synthesizer
Provides rich, authoritative medical explanations regarding cancer biology,
types, symptoms, staging, therapies, prevention, genetics, and diagnostics.
All responses are formatted with clean bullet points and clear clinical headings.
"""

import re
from typing import Dict, Any, Optional, List

# ── Structured Oncology Topics ───────────────────────────────────────────────
ONCOLOGY_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "what_is_cancer": {
        "title": "Understanding What Cancer Is",
        "content": (
            "### 🧬 What is Cancer?\n"
            "**Cancer** is a group of over 200 diseases characterized by the **uncontrolled growth, division, and spread of abnormal cells** in the body. In a healthy organism, cells grow, divide, and naturally die (apoptosis) in an orderly cycle. When genetic mutations disrupt this delicate process, damaged or mutated cells survive, multiply uncontrollably, and can invade surrounding tissues or travel to distant organs.\n\n"
            "#### 🔬 Biological Hallmarks of Cancer:\n"
            "• **Sustained Proliferative Signaling**: Cancer cells continuously produce growth signals or mutate receptors to multiply non-stop.\n"
            "• **Evading Growth Suppressors**: Inactivation of tumor-suppressor genes (like TP53 and RB1) prevents the body from halting abnormal cell division.\n"
            "• **Resisting Cell Death (Apoptosis)**: Cancer cells bypass the programmed self-destruction mechanism that normally eliminates damaged cells.\n"
            "• **Enabling Replicative Immortality**: Activation of telomerase prevents chromosome degradation, allowing endless cell replications.\n"
            "• **Inducing Angiogenesis**: Tumors stimulate the growth of new capillary blood vessels to supply themselves with oxygen and nutrients.\n"
            "• **Activating Invasion & Metastasis**: Cancer cells detach from the primary tumor, enter the bloodstream or lymphatic system, and establish secondary tumors in other organs.\n\n"
            "#### ⚖️ Benign vs. Malignant Tumors:\n"
            "• **Benign (Non-Cancerous)**:\n"
            "  - Slow-growing, encapsulated, and clearly defined boundaries.\n"
            "  - Do not invade adjacent healthy tissues or spread to distant organs.\n"
            "  - Usually non-life-threatening (unless located in critical areas like the brain).\n"
            "• **Malignant (Cancerous)**:\n"
            "  - Fast-growing, irregular margins, and non-encapsulated.\n"
            "  - Invade and destroy surrounding normal tissue.\n"
            "  - Capable of metastasizing (spreading) throughout the body via blood and lymph.\n\n"
            "#### 📊 Main Categories of Cancer by Tissue Origin:\n"
            "• **Carcinomas**: Arise in epithelial tissues covering external body surfaces and internal organ linings (~80–90% of all cancers; e.g., breast, lung, prostate, colon).\n"
            "• **Sarcomas**: Originate in supportive and connective tissues (bone, cartilage, fat, muscle, and blood vessels).\n"
            "• **Leukemias**: Malignancies of blood-forming bone marrow tissue, causing abnormal accumulation of immature white blood cells.\n"
            "• **Lymphomas**: Cancers originating in lymphocytes and lymph node structures of the immune system.\n"
            "• **Central Nervous System (CNS) Cancers**: Tumors that develop in the tissues of the brain and spinal cord (e.g., Gliomas, Glioblastomas).\n\n"
            "#### ⚠️ Primary Causes & Risk Factors:\n"
            "• **Genetic Factors**: Inherited DNA mutations (e.g., BRCA1/2, Lynch syndrome) or acquired somatic mutations.\n"
            "• **Chemical Carcinogens**: Tobacco smoke, asbestos, benzene, and occupational chemicals.\n"
            "• **Radiation**: Ultraviolet (UV) rays from sunlight/tanning beds and ionizing radiation.\n"
            "• **Infectious Agents**: Human Papillomavirus (HPV), Hepatitis B & C viruses, Helicobacter pylori, and Epstein-Barr virus (EBV).\n"
            "• **Lifestyle Factors**: Obesity, sedentary lifestyle, excessive alcohol consumption, and chronic inflammation."
        )
    },
    "symptoms_warning_signs": {
        "title": "Early Warning Signs & Symptoms of Cancer",
        "content": (
            "### ⚠️ Early Warning Signs of Cancer\n"
            "Recognizing symptoms early significantly improves treatment success. The American Cancer Society and WHO highlight the classic **C-A-U-T-I-O-N** indicators:\n\n"
            "• **C** - Change in bowel or bladder habits (e.g., persistent diarrhea, constipation, or blood in urine/stool).\n"
            "• **A** - A sore or ulcer that does not heal after 2-3 weeks.\n"
            "• **U** - Unusual bleeding or discharge (e.g., coughing up blood, abnormal vaginal bleeding, blood in sputum).\n"
            "• **T** - Thickening or lump in the breast, testicles, lymph nodes, neck, or soft tissues.\n"
            "• **I** - Indigestion or chronic difficulty swallowing (dysphagia).\n"
            "• **O** - Obvious change in the size, color, or shape of a skin wart or mole (ABCDE rule).\n"
            "• **N** - Nagging, persistent cough or chronic hoarseness.\n\n"
            "#### 🩺 General Systemic Symptoms:\n"
            "• **Unexplained Weight Loss**: Losing 10+ lbs (4.5+ kg) rapidly without changes in diet or physical exercise.\n"
            "• **Persistent Fatigue**: Severe, debilitating exhaustion that does not improve after rest or sleep.\n"
            "• **Unexplained Recurrent Fevers & Night Sweats**: Common presenting signs in lymphomas and leukemias.\n"
            "• **Unexplained Chronic Pain**: Bone pain, persistent localized abdominal pain, or morning headaches with vomiting.\n"
            "• **Swollen Lymph Nodes**: Painless, firm swelling in the neck, armpit, or groin persisting >2-3 weeks.\n\n"
            "*Note: Having one or more of these symptoms does not confirm cancer, but persistent symptoms warrant prompt clinical evaluation by a physician.*"
        )
    },
    "lung_cancer": {
        "title": "Lung Cancer (Symptoms, Types, Screening, Treatment)",
        "content": (
            "### 🫁 Comprehensive Overview: Lung Cancer\n"
            "Lung cancer is the leading cause of cancer-related mortality globally. Early detection through low-dose CT scans dramatically improves 5-year survival rates.\n\n"
            "#### 📑 Major Subtypes:\n"
            "• **Non-Small Cell Lung Cancer (NSCLC)** (~85% of cases):\n"
            "  - *Adenocarcinoma*: Most common in non-smokers and women; arises in outer mucus-producing lung regions.\n"
            "  - *Squamous Cell Carcinoma*: Strongly linked to smoking history; originates in central airway passages.\n"
            "  - *Large Cell Carcinoma*: Rapidly growing, undifferentiated aggressive tumor.\n"
            "• **Small Cell Lung Cancer (SCLC)** (~15% of cases):\n"
            "  - Highly aggressive neuroendocrine tumor strongly linked to heavy tobacco smoke exposure; tends to metastasize early.\n\n"
            "#### ⚠️ Key Warning Signs & Symptoms:\n"
            "• Chronic, worsening cough lasting longer than 3 weeks.\n"
            "• **Hemoptysis**: Coughing up blood or rust-streaked phlegm.\n"
            "• Progressive shortness of breath, unexplained wheezing, or chest tightness.\n"
            "• Dull or sharp chest, shoulder, or back pain exacerbated by coughing or deep breathing.\n"
            "• Persistent hoarseness due to recurrent laryngeal nerve involvement.\n"
            "• Recurrent episodes of bronchitis or pneumonia in the same lung zone.\n\n"
            "#### 🔍 Screening & Diagnostics:\n"
            "• **Low-Dose Computed Tomography (LDCT)**: Annual screening recommended for adults aged 50–80 with a 20+ pack-year smoking history.\n"
            "• **Biopsy & Biomarker Testing**: Bronchoscopy, CT-guided needle biopsy, and NGS testing for EGFR, ALK, ROS1, KRAS, and PD-L1 expression.\n\n"
            "#### 💊 Standard Treatment Approaches:\n"
            "• **Surgery**: Lobectomy, segmentectomy, or pneumonectomy for localized stages.\n"
            "• **Targeted Therapy**: Tyrosine kinase inhibitors (Osimertinib, Alectinib) for specific genetic mutations.\n"
            "• **Immunotherapy**: Checkpoint inhibitors (Pembrolizumab, Nivolumab) targeting PD-1/PD-L1.\n"
            "• **Stereotactic Radiation & Chemotherapy**: SBRT for inoperable nodules or platinum-based chemotherapy combinations."
        )
    },
    "brain_tumor": {
        "title": "Brain Tumors & Neuro-Oncology",
        "content": (
            "### 🧠 Comprehensive Overview: Brain Tumors\n"
            "Brain tumors originate within intracranial tissues or spread from distant cancers (metastases). They are categorized by cell origin and WHO malignancy grade (I-IV).\n\n"
            "#### 📑 Primary Brain Tumor Types:\n"
            "• **Gliomas** (Astrocytoma, Oligodendroglioma, Ependymoma):\n"
            "  - *Glioblastoma (GBM)*: Grade IV astrocytoma; the most aggressive and common malignant brain tumor in adults.\n"
            "• **Meningioma**:\n"
            "  - Originates in the protective meningeal membranes surrounding brain and spinal cord; ~80-90% are benign (Grade I).\n"
            "• **Pituitary Adenoma**:\n"
            "  - Develops in the pituitary gland; can cause endocrine hormonal disruptions and optic chiasm compression (vision loss).\n"
            "• **Brain Metastases**:\n"
            "  - Secondary tumors spreading most commonly from lung, breast, melanoma, renal cell, or colorectal cancers.\n\n"
            "#### ⚠️ Hallmark Symptoms:\n"
            "• Persistent, worsening headaches that are most severe upon waking up in the morning and aggravated by coughing or bending over.\n"
            "• **New-Onset Seizures**: Convulsions or focal sensory episodes in individuals without prior history of epilepsy.\n"
            "• Cognitive & Personality Changes: Memory deficits, mood swings, impaired concentration, or confusion.\n"
            "• Focal Neurological Deficits: Hemiparesis (weakness/numbness on one side of the body), balance loss (ataxia), or speech difficulty (aphasia).\n"
            "• Visual Disturbances: Double vision (diplopia), blurred vision, or loss of peripheral vision.\n"
            "• Morning Nausea and Projectile Vomiting without preceding gastrointestinal illness.\n\n"
            "#### 🔍 Diagnostics & Treatments:\n"
            "• **Magnetic Resonance Imaging (MRI)** with intravenous Gadolinium contrast (T1, T2, FLAIR sequences).\n"
            "• **Microsurgical Craniotomy**: Maximal safe resection guided by intraoperative neuronavigation.\n"
            "• **Radiation**: Intensity-Modulated Radiation Therapy (IMRT) and Stereotactic Radiosurgery (Gamma Knife / CyberKnife).\n"
            "• **Chemotherapy**: Oral Temozolomide (crosses blood-brain barrier) and Tumor Treating Fields (TTFields)."
        )
    },
    "breast_cancer": {
        "title": "Breast Cancer (Subtypes, Screening, Treatments)",
        "content": (
            "### 🎗️ Comprehensive Overview: Breast Cancer\n"
            "Breast cancer starts in the milk ducts (ductal) or lobules (lobular) of breast tissue. It is the most frequently diagnosed cancer in women globally.\n\n"
            "#### 📑 Molecular Subtypes:\n"
            "• **HR+ / HER2- (Luminal A/B)**: Estrogen receptor (ER) or progesterone receptor (PR) positive; most common and responds well to hormone therapy.\n"
            "• **HER2-Positive**: Overexpresses the HER2 receptor tyrosine kinase; sensitive to targeted anti-HER2 therapies (e.g., Trastuzumab/Herceptin, Pertuzumab).\n"
            "• **Triple-Negative Breast Cancer (TNBC)**: Lacks ER, PR, and HER2 receptors; treated primarily with cytotoxic chemotherapy and immunotherapy.\n\n"
            "#### ⚠️ Hallmark Symptoms:\n"
            "• A hard, painless, non-tender lump or mass with irregular borders in the breast or axillary (armpit) lymph nodes.\n"
            "• Noticeable alteration in breast shape, contour, symmetry, or size.\n"
            "• **Skin Changes**: Dimpling, puckering, erythema (redness), or orange-peel texture (*peau d'orange*).\n"
            "• **Nipple Abnormalities**: Newly inverted nipple, persistent scaling/crusting, or spontaneous bloody/clear nipple discharge.\n\n"
            "#### 🔍 Screening & Genetics:\n"
            "• **Screening Mammography**: Recommended annually for average-risk women starting at age 40 to 45.\n"
            "• **Breast MRI & Ultrasound**: Supplemental imaging for women with dense breast tissue or high familial risk.\n"
            "• **Genetic Testing**: BRCA1 and BRCA2 germline mutation screening for individuals with strong family history.\n\n"
            "#### 💊 Standard Treatments:\n"
            "• **Surgical**: Breast-conserving lumpectomy (with adjuvant radiotherapy) or total mastectomy with sentinel lymph node evaluation.\n"
            "• **Hormone Therapy**: Selective Estrogen Receptor Modulators (Tamoxifen) or Aromatase Inhibitors (Letrozole, Anastrozole).\n"
            "• **Chemotherapy & Targeted Agents**: CDK4/6 inhibitors (Palbociclib, Ribociclib) and PARP inhibitors (Olaparib)."
        )
    },
    "skin_cancer": {
        "title": "Skin Cancer & Melanoma Detection",
        "content": (
            "### 🔬 Comprehensive Overview: Skin Cancer & Melanoma\n"
            "Skin cancer arises primarily from epidermal damage caused by cumulative ultraviolet (UV) radiation from sun exposure or artificial tanning devices.\n\n"
            "#### 📑 Three Primary Types:\n"
            "• **Basal Cell Carcinoma (BCC)** (~80% of cases): Pearly translucent papule, pink patch, or non-healing bleeding sore; very rarely spreads internally but locally destructive.\n"
            "• **Squamous Cell Carcinoma (SCC)** (~15% of cases): Firm red nodule, scaly crusted lesion, or ulcerated growth; moderate risk of regional lymph node spread.\n"
            "• **Melanoma** (~5% of cases): Highly dangerous malignancy of pigment-producing melanocytes; responsible for the majority of skin cancer fatalities if not caught early.\n\n"
            "#### 🔍 The ABCDE Rule for Melanoma Detection:\n"
            "• **A - Asymmetry**: One half of the mole or lesion does not match the other half in shape.\n"
            "• **B - Border**: The edges are uneven, ragged, notched, scalloped, or poorly defined.\n"
            "• **C - Color**: Presence of multiple varying shades of tan, dark brown, black, white, red, or blue.\n"
            "• **D - Diameter**: Larger than 6 millimeters across (approx. diameter of a standard pencil eraser).\n"
            "• **E - Evolving**: Any perceptible change in size, shape, surface elevation, itching, bleeding, or crusting over time.\n\n"
            "#### 🛡️ Prevention & Treatments:\n"
            "• **UV Protection**: Use broad-spectrum SPF 30+ sunscreen, wear UV-rated protective apparel, and avoid peak sun (10 AM – 4 PM).\n"
            "• **Therapies**: Wide local excision, Mohs micrographic surgery, targeted BRAF/MEK inhibitors (Dabrafenib/Trametinib), and immunotherapy (Pembrolizumab/Ipilimumab)."
        )
    },
    "colorectal_cancer": {
        "title": "Colorectal Cancer (Colon & Rectal Cancer)",
        "content": (
            "### 🩺 Comprehensive Overview: Colorectal Cancer\n"
            "Colorectal cancer originates in the inner lining of the large intestine (colon) or rectum, typically developing gradually from benign adenomatous polyps over 10–15 years.\n\n"
            "#### ⚠️ Hallmark Symptoms:\n"
            "• **Rectal Bleeding**: Bright red blood in stool, or dark, maroon, tarry stools (melena).\n"
            "• **Altered Bowel Habits**: Persistent diarrhea, severe constipation, or narrowing of stool caliber lasting >3-4 weeks.\n"
            "• **Abdominal Symptoms**: Persistent gas, bloating, painful cramps, or feeling of incomplete bowel evacuation (tenesmus).\n"
            "• **Systemic Signs**: Unexplained iron-deficiency anemia, chronic fatigue, and unintentional weight loss.\n\n"
            "#### 🔍 Screening & Prevention:\n"
            "• **Screening Colonoscopy**: Routine screening begins at age 45; enables simultaneous visual detection and complete polyp removal (polypectomy).\n"
            "• **Non-Invasive Tests**: Fecal Immunochemical Test (FIT) annually or stool DNA-FIT test (Cologuard) every 3 years.\n"
            "• **Dietary Risk Reduction**: High-fiber diet (vegetables, whole grains, legumes), avoiding processed meats (bacon, sausages), and limiting alcohol.\n\n"
            "#### 💊 Standard Treatments:\n"
            "• **Surgical Resection**: Colectomy (removal of affected bowel segment) with regional mesenteric lymphadenectomy.\n"
            "• **Chemotherapy Regimens**: FOLFOX or FOLFIRI regimens for nodal involvement (Stage III).\n"
            "• **Targeted Biologics & Immunotherapy**: Anti-VEGF (Bevacizumab), anti-EGFR (Cetuximab), and checkpoint inhibitors for MSI-High tumors."
        )
    },
    "blood_cancer": {
        "title": "Blood Cancers: Leukemia, Lymphoma & Multiple Myeloma",
        "content": (
            "### 🩸 Comprehensive Overview: Blood Cancers (Hematologic Malignancies)\n"
            "Hematologic malignancies affect blood cell formation in the bone marrow, circulation, and the lymphatic system.\n\n"
            "#### 📑 1. Leukemia (Bone Marrow Malignancy):\n"
            "• **Acute Lymphoblastic Leukemia (ALL)**: Fast-progressing; most common childhood leukemia; high cure rate with chemotherapy protocols.\n"
            "• **Acute Myeloid Leukemia (AML)**: Rapid accumulation of abnormal blast cells; common in older adults.\n"
            "• **Chronic Leukemias (CLL & CML)**: Slower-progressing; CML is characterized by the Philadelphia chromosome (BCR-ABL fusion gene).\n"
            "• **Hallmark Symptoms**: Severe fatigue (anemia), easy bleeding/bruising (thrombocytopenia), petechiae (pinpoint red skin spots), and frequent infections.\n\n"
            "#### 📑 2. Lymphoma (Lymphatic Malignancy):\n"
            "• **Hodgkin Lymphoma (HL)**: Characterized microscopically by giant Reed-Sternberg cells; highly treatable with chemotherapy and radiation.\n"
            "• **Non-Hodgkin Lymphoma (NHL)**: Broad class including Diffuse Large B-Cell Lymphoma (DLBCL) and Follicular Lymphoma.\n"
            "• **Hallmark Symptoms**: Painless swelling of lymph nodes in the neck, armpits, or groin, accompanied by 'B-Symptoms' (fevers >38°C, drenching night sweats, >10% weight loss).\n\n"
            "#### 📑 3. Multiple Myeloma:\n"
            "• Malignant proliferation of plasma cells producing abnormal monoclonal paraprotein (M-protein).\n"
            "• **CRAB Criteria**: Hyper**C**alcemia, **R**enal insufficiency, **A**nemia, and osteolytic **B**one pain/fractures.\n\n"
            "#### 💊 Advanced Therapies:\n"
            "• Targeted Tyrosine Kinase Inhibitors (Imatinib / Gleevec for CML).\n"
            "• **Autologous / Allogeneic Stem Cell & Bone Marrow Transplants**.\n"
            "• **CAR-T Cell Therapy**: Re-engineering patient T-cells to target CD19 or BCMA surface antigens."
        )
    },
    "prostate_cancer": {
        "title": "Prostate Cancer (Symptoms, Screening, Treatments)",
        "content": (
            "### 🩺 Comprehensive Overview: Prostate Cancer\n"
            "Prostate cancer develops in the small walnut-shaped gland that produces seminal fluid in men. It is one of the most common cancers in men globally.\n\n"
            "#### ⚠️ Warning Signs & Symptoms:\n"
            "• Early stages are frequently asymptomatic and detected solely via screening.\n"
            "• Difficulty initiating urination or weak, interrupted urinary stream.\n"
            "• Increased frequency of urination, especially waking frequently at night (nocturia).\n"
            "• **Hematuria**: Blood in urine or semen.\n"
            "• Pelvic, hip, or lower back discomfort when advanced or metastasized to bones.\n"
            "• Erectile dysfunction or painful ejaculation.\n\n"
            "#### 🔍 Screening & Grading:\n"
            "• **PSA Blood Test**: Prostate-Specific Antigen blood test; values >4 ng/mL warrant clinical follow-up.\n"
            "• **Digital Rectal Exam (DRE)**: Physical examination of prostate gland contour and consistency.\n"
            "• **Gleason Score / Grade Group**: Biopsy grading system (scores 6 to 10) evaluating tumor aggressiveness.\n\n"
            "#### 💊 Standard Treatments:\n"
            "• **Active Surveillance**: Close monitoring with PSA and repeat biopsies for low-risk, slow-growing tumors.\n"
            "• **Radical Prostatectomy**: Surgical removal of prostate gland and seminal vesicles.\n"
            "• **Radiation**: External Beam Radiotherapy (EBRT) or Brachytherapy (radioactive seed implants).\n"
            "• **Androgen Deprivation Therapy (ADT)**: Hormone therapy suppressing testosterone to stop cancer growth."
        )
    },
    "cervical_cancer": {
        "title": "Cervical Cancer & HPV Prevention",
        "content": (
            "### 🌸 Comprehensive Overview: Cervical Cancer\n"
            "Cervical cancer occurs in the cells of the cervix (the lower part of the uterus connecting to the vagina). Nearly 99% of cases are caused by persistent infection with high-risk **Human Papillomavirus (HPV)** strains (HPV-16 and HPV-18).\n\n"
            "#### ⚠️ Hallmark Symptoms:\n"
            "• Abnormal vaginal bleeding between menstrual cycles, after sexual intercourse, or after menopause.\n"
            "• Unusual, persistent, foul-smelling or watery vaginal discharge.\n"
            "• Deep pelvic pain or discomfort during intercourse (dyspareunia).\n"
            "• Back, leg, or pelvic pain in advanced stages.\n\n"
            "#### 🛡️ Prevention & Routine Screening:\n"
            "• **HPV Vaccine (Gardasil 9)**: Highly effective preventive immunization recommended for adolescents and adults aged 9 to 26 (and up to 45 with clinical discussion); prevents up to 90% of cervical cancers.\n"
            "• **Pap Smear (Cytology)**: Recommended every 3 years for women aged 21–65 to identify precancerous cellular dysplasias.\n"
            "• **HPV High-Risk DNA Test**: Recommended every 5 years for women aged 30–65.\n\n"
            "#### 💊 Standard Treatments:\n"
            "• **Precancerous Lesions**: LEEP (Loop Electrosurgical Excision Procedure) or Cold Knife Conization.\n"
            "• **Invasive Stages**: Total hysterectomy, concurrent chemoradiation (Cisplatin + Radiation), and anti-PD-1 immunotherapy."
        )
    },
    "pancreatic_cancer": {
        "title": "Pancreatic Cancer (Symptoms, Risks, Treatment)",
        "content": (
            "### 🩺 Comprehensive Overview: Pancreatic Cancer\n"
            "Pancreatic adenocarcinoma arises predominantly in the exocrine ducts of the pancreas. Because the pancreas lies deep in the retroperitoneum, early tumors rarely cause noticeable symptoms.\n\n"
            "#### ⚠️ Warning Signs & Symptoms:\n"
            "• **Painless Jaundice**: Yellowing of the skin and whites of the eyes, accompanied by dark tea-colored urine and pale clay-colored stools.\n"
            "• **Abdominal & Mid-Back Pain**: Dull ache radiating from the upper abdomen to the middle back.\n"
            "• **Unexplained Rapid Weight Loss** and significant loss of appetite (anorexia).\n"
            "• **New-Onset Diabetes**: Sudden onset of diabetes in an older adult without family history or obesity.\n"
            "• Severe pruritus (generalized skin itching due to bile salt buildup).\n\n"
            "#### 🔍 Diagnostics & Treatments:\n"
            "• **High-Resolution Multi-Phase CT / MRI** of the pancreas and Endoscopic Ultrasound (EUS) with FNA biopsy.\n"
            "• **Whipple Procedure (Pancreaticoduodenectomy)**: Surgical resection of the pancreatic head, duodenum, and gallbladder for resectable tumors.\n"
            "• **Systemic Chemotherapy**: FOLFIRINOX or Gemcitabine + nab-Paclitaxel combinations."
        )
    },
    "cancer_staging": {
        "title": "Cancer Staging & Grading (TNM System)",
        "content": (
            "### 📊 Cancer Staging & Grading Explained\n"
            "Staging defines the exact **size of the primary tumor and how far cancer cells have spread** across the body. Staging determines patient prognosis and optimal treatment protocols.\n\n"
            "#### 📐 The TNM Staging Framework (AJCC/UICC):\n"
            "• **T (Tumor)**: Dimensions and local invasiveness of the primary tumor (Tis = in situ, T1 to T4 = increasing size/depth).\n"
            "• **N (Nodes)**: Spread to regional draining lymph nodes (N0 = no node spread, N1–N3 = progressive lymph node involvement).\n"
            "• **M (Metastasis)**: Spread to distant organs such as liver, lungs, brain, or bones (M0 = no distant spread, M1 = distant metastasis).\n\n"
            "#### 🔢 Overall Stage Groupings (0 Through IV):\n"
            "• **Stage 0 (Carcinoma in Situ)**: Pre-cancerous abnormal cells confined to the initial epithelial layer; non-invasive and highly curable.\n"
            "• **Stage I (Early Localized)**: Small tumor confined completely to the organ of origin with no nodal involvement; excellent prognosis.\n"
            "• **Stage II (Locally Invasive)**: Larger primary tumor that has grown deeply into adjacent tissues with minimal nodal spread.\n"
            "• **Stage III (Regional Nodal Spread)**: Large tumor with extensive regional lymph node metastases; treated with multimodal therapy (surgery + chemo + radiation).\n"
            "• **Stage IV (Metastatic / Advanced)**: Cancer has spread to distant organs and tissues; managed primarily with systemic therapies (chemo, immunotherapy, targeted therapy).\n\n"
            "#### 🔬 Histopathological Tumor Grading (Differentiation):\n"
            "• **Grade 1 (Well-Differentiated)**: Tumor cells closely resemble normal cells; slow-growing.\n"
            "• **Grade 2 (Moderately Differentiated)**: Intermediate cellular abnormality and growth rate.\n"
            "• **Grade 3/4 (Poorly Differentiated / Anaplastic)**: Highly abnormal, disorganized cells; rapidly growing and invasive."
        )
    },
    "cancer_treatments": {
        "title": "Oncology Treatments (Chemo, Radiation, Immunotherapy, Surgery)",
        "content": (
            "### 💊 Overview of Modern Cancer Treatments\n"
            "Modern oncology uses multimodal combinations tailored to tumor biology, genomic alterations, and individual patient health.\n\n"
            "#### 🔪 1. Surgical Oncology:\n"
            "• Complete surgical removal of the primary tumor mass with clear microscopic margins (R0 resection) and sentinel lymph nodes. Curative in localized stages.\n\n"
            "#### 🧪 2. Chemotherapy:\n"
            "• Systemic cytotoxic drugs (Cisplatin, Carboplatin, Paclitaxel, Doxorubicin, 5-FU) that disrupt cell division and induce apoptosis in fast-replicating cells.\n\n"
            "#### ⚡ 3. Radiation Therapy:\n"
            "• Targeted ionizing radiation (X-rays, gamma rays, proton beams) that shatters cancer cell DNA.\n"
            "• Delivered via **External Beam (IMRT/SBRT)** or internally via **Brachytherapy** radioactive seeds.\n\n"
            "#### 🛡️ 4. Immunotherapy (Immune Checkpoint Inhibitors):\n"
            "• Re-activates the patient's own immune system to recognize and eliminate tumor cells.\n"
            "• Anti-PD-1 (Pembrolizumab, Nivolumab), Anti-PD-L1 (Atezolizumab), Anti-CTLA-4 (Ipilimumab), and **CAR-T Cell Therapy**.\n\n"
            "#### 🎯 5. Targeted Precision Therapy:\n"
            "• Engineered small molecules or monoclonal antibodies that specifically neutralize mutated oncogenic proteins:\n"
            "  - *EGFR / ALK Inhibitors* (Osimertinib, Alectinib for lung cancer)\n"
            "  - *HER2 Inhibitors* (Trastuzumab, Pertuzumab for breast/gastric cancer)\n"
            "  - *BRAF / MEK Inhibitors* (Dabrafenib, Trametinib for melanoma)\n"
            "  - *PARP Inhibitors* (Olaparib for BRCA-mutated ovarian/breast cancer)\n\n"
            "#### 🧬 6. Hormone (Endocrine) Therapy:\n"
            "• Blocks hormones that fuel hormone-dependent cancers (Tamoxifen, Aromatase Inhibitors for breast cancer; ADT for prostate cancer)."
        )
    },
    "chemotherapy_side_effects": {
        "title": "Chemotherapy Side Effects & Management",
        "content": (
            "### 🛡️ Chemotherapy Side Effects & Practical Management\n"
            "Chemotherapy eliminates rapidly dividing cells. Because healthy cells in the bone marrow, hair follicles, and digestive tract also divide quickly, temporary side effects occur:\n\n"
            "#### 🩺 Key Side Effects & Relief Strategies:\n"
            "• **Cancer-Related Fatigue**: The most common side effect. Prioritize rest, take short 20-minute naps, and engage in gentle daily walks.\n"
            "• **Nausea & Vomiting**: Highly manageable with modern 5-HT3 and NK1 antiemetic drugs (Ondansetron, Aprepitant). Eat small, frequent, bland meals (crackers, toast, ginger tea).\n"
            "• **Neutropenia & Infection Risk**: Low white blood cell counts leave patients vulnerable. Wash hands frequently, avoid raw foods and crowded indoor spaces, and monitor temperature regularly.\n"
            "• **Hair Loss (Alopecia)**: Begins 2–3 weeks after starting therapy; hair regrows 2–3 months post-treatment. Scalp cooling caps (cold caps) significantly reduce hair loss.\n"
            "• **Mouth Sores (Mucositis)**: Rinse mouth several times daily with warm salt/baking soda water; avoid acidic, citrus, or spicy foods.\n"
            "• **Peripheral Neuropathy**: Tingling, numbness, or burning sensation in fingers and toes; report immediately to your oncologist for dosage adjustments.\n\n"
            "*🚨 Medical Alert: Seek emergency medical care immediately if you develop a fever ≥100.4°F (38.0°C), severe chills, or shortness of breath while on chemotherapy.*"
        )
    },
    "cancer_prevention": {
        "title": "Cancer Prevention & Risk Reduction Guidelines",
        "content": (
            "### 🛡️ Evidence-Based Cancer Prevention Guidelines\n"
            "According to the World Health Organization (WHO) and National Cancer Institute (NCI), **30% to 50% of all cancers are entirely preventable** through lifestyle choices, immunization, and environmental precautions:\n\n"
            "#### 1. Complete Tobacco & Nicotine Avoidance:\n"
            "• Tobacco use is responsible for ~85% of lung cancers and causes oral, esophageal, bladder, and pancreatic malignancies. Quitting immediately begins healing respiratory tissues.\n\n"
            "#### 2. Healthy Weight & Regular Exercise:\n"
            "• Obesity is directly linked to 13+ distinct cancers (colorectal, postmenopausal breast, uterine, kidney). Engage in at least 150 minutes of moderate aerobic activity per week.\n\n"
            "#### 3. Nutrient-Dense, Plant-Forward Diet:\n"
            "• Consume abundant cruciferous vegetables (broccoli, cauliflower), dark leafy greens, berries, legumes, and whole grains.\n"
            "• Avoid processed meats (bacon, cold cuts, hot dogs) and limit red meat to less than 3 portions weekly.\n\n"
            "#### 4. Limit or Eliminate Alcohol Consumption:\n"
            "• Alcohol is classified as a Group 1 carcinogen linked to breast, colorectal, esophageal, and liver cancers.\n\n"
            "#### 5. Rigorous Sun & UV Protection:\n"
            "• Apply broad-spectrum SPF 30+ sunscreen every 2 hours, wear UV sunglasses, and avoid indoor UV tanning beds.\n\n"
            "#### 6. Preventive Vaccinations:\n"
            "• **HPV Vaccine**: Prevents up to 90% of cervical, anal, vaginal, and oropharyngeal cancers.\n"
            "• **Hepatitis B Vaccine**: Prevents chronic HBV infection and associated primary liver cancer (hepatocellular carcinoma).\n\n"
            "#### 7. Routine Age-Appropriate Screenings:\n"
            "• Regular mammograms, colonoscopies, Pap smears, and low-dose CT scans detect pre-malignant changes when treatment is most curative."
        )
    },
    "diagnostics_screening": {
        "title": "Cancer Diagnostics, Biopsies & Screening Modalities",
        "content": (
            "### 🔍 How Cancer is Detected, Screened & Diagnosed\n"
            "Definitive oncology diagnosis requires histological confirmation combined with state-of-the-art diagnostic imaging:\n\n"
            "#### 🔬 1. Tissue Biopsy (Definitive Gold Standard):\n"
            "• **Fine-Needle Aspiration (FNA)**: Uses a thin needle to withdraw fluid and cellular clusters for cytological review.\n"
            "• **Core Needle Biopsy**: Extracts cylindrical tissue cores preserving histological architecture and receptor status.\n"
            "• **Excisional / Surgical Biopsy**: Removes the entire suspicious nodule or lymph node for microscopic analysis.\n\n"
            "#### 📡 2. Advanced Diagnostic Imaging:\n"
            "• **Computed Tomography (CT)**: Rapid 3D cross-sectional imaging defining tumor dimensions and lymph node enlargement.\n"
            "• **Magnetic Resonance Imaging (MRI)**: Superior soft-tissue contrast; critical for neuro-oncology, spinal cord, and musculoskeletal tumors.\n"
            "• **PET-CT Scans**: Uses radioactive fluorodeoxyglucose (FDG) to map hyper-metabolic, glucose-consuming malignant metastases throughout the entire body.\n"
            "• **Mammography & Ultrasound**: High-frequency sound waves and low-dose ionizing radiation for breast, thyroid, and pelvic evaluations.\n\n"
            "#### 🩸 3. Laboratory Biomarkers & Liquid Biopsies:\n"
            "• **Serum Tumor Markers**: PSA (prostate), CA-125 (ovarian), CEA (colorectal), AFP (liver), and CA 19-9 (pancreatic).\n"
            "• **Next-Generation Genomic Sequencing (NGS)**: Identifies actionable driver mutations in tumor DNA or circulating tumor DNA (ctDNA)."
        )
    },
    "genetics_hereditary": {
        "title": "Hereditary Cancer Genetics & Gene Mutations",
        "content": (
            "### 🧬 Hereditary Cancer Syndromes & Genetic Mutations\n"
            "Approximately 5% to 10% of all cancers are hereditary, caused by inherited pathogenic germline gene mutations passed from parents to children.\n\n"
            "#### 📑 Key Hereditary Cancer Genes & Syndromes:\n"
            "• **BRCA1 and BRCA2 Mutations**:\n"
            "  - Causes **Hereditary Breast and Ovarian Cancer (HBOC) syndrome**.\n"
            "  - Lifetime risk of breast cancer up to 72% (BRCA1) and 69% (BRCA2); also elevates risk of ovarian, pancreatic, and prostate cancers.\n"
            "• **Lynch Syndrome (HNPCC)**:\n"
            "  - Caused by DNA mismatch repair (MMR) gene mutations (MLH1, MSH2, MSH6, PMS2).\n"
            "  - Greatly increases lifetime risk of colorectal cancer (~70-80%) and endometrial/uterine cancer.\n"
            "• **Li-Fraumeni Syndrome**:\n"
            "  - Caused by inherited TP53 tumor suppressor mutations; causes early-onset sarcomas, breast cancer, brain tumors, and leukemias.\n"
            "• **Familial Adenomatous Polyposis (FAP)**:\n"
            "  - Caused by APC gene mutations; leads to hundreds of precancerous polyps in the colon by age 20-30.\n\n"
            "#### 🔍 Genetic Counseling & Testing:\n"
            "• Recommended for individuals with multiple close relatives diagnosed with cancer, cancer diagnosed before age 50, or rare tumors.\n"
            "• Enables proactive preventive measures: enhanced screening (breast MRI, early colonoscopy) or risk-reducing preventive surgeries."
        )
    },
    "hospital_system_tools": {
        "title": "Hospital Clinical AI Detection Capabilities",
        "content": (
            "### 🏥 Hospital Clinical AI Diagnostic Support\n"
            "Our Erode District Clinical AI platform provides cutting-edge automated deep-learning models to assist clinicians and patients:\n\n"
            "• **🧠 Brain Tumor Classifier**: Vision Transformer (ViT) & ResNet50 models for MRI & CT scans, classifying Glioma, Meningioma, Pituitary, or Healthy Brain.\n"
            "• **🫁 Lung Cancer Detector**: ResNet50 convolutional network analyzing Chest CT images for malignant vs benign pulmonary nodules.\n"
            "• **🔬 Skin Lesion Analyzer**: EfficientNet architecture trained on dermoscopy images for early Melanoma vs Benign Nevi discrimination.\n"
            "• **🔥 Explainable AI (Grad-CAM)**: Generates localized visual heatmaps showing the exact radiographic regions influencing the neural network prediction.\n\n"
            "To use these tools, navigate to **Upload Image & Analyze** in the top menu."
        )
    }
}


def find_best_knowledge_match(query: str) -> Optional[Dict[str, Any]]:
    """
    Match user query to the most relevant clinical topic based on semantic keyword density.
    """
    q = query.lower()

    # Priority 1: Specific cancer types
    if any(k in q for k in ["lung", "respiratory", "coughing blood", "hemoptysis", "pulmonary nodule", "sclc", "nsclc"]):
        return ONCOLOGY_KNOWLEDGE["lung_cancer"]
    if any(k in q for k in ["brain", "glioma", "glioblastoma", "meningioma", "pituitary", "headache and seizure", "craniotomy"]):
        return ONCOLOGY_KNOWLEDGE["brain_tumor"]
    if any(k in q for k in ["skin", "melanoma", "mole", "abcde", "basal cell", "squamous cell", "sunburn", "dermoscop"]):
        return ONCOLOGY_KNOWLEDGE["skin_cancer"]
    if any(k in q for k in ["breast", "mammogra", "mastectomy", "lumpectomy", "her2", "brca", "nipple discharge"]):
        return ONCOLOGY_KNOWLEDGE["breast_cancer"]
    if any(k in q for k in ["colon", "rectal", "colorectal", "polyps", "colonoscop", "blood in stool", "tenesmus"]):
        return ONCOLOGY_KNOWLEDGE["colorectal_cancer"]
    if any(k in q for k in ["prostate", "psa", "urinary stream", "gleason", "prostatectomy", "nocturia"]):
        return ONCOLOGY_KNOWLEDGE["prostate_cancer"]
    if any(k in q for k in ["cervical", "cervix", "hpv", "pap smear", "gardasil", "vaginal bleeding"]):
        return ONCOLOGY_KNOWLEDGE["cervical_cancer"]
    if any(k in q for k in ["pancrea", "jaundice", "whipple", "pale stool", "dark urine"]):
        return ONCOLOGY_KNOWLEDGE["pancreatic_cancer"]
    if any(k in q for k in ["leukemia", "leukaemia", "lymphoma", "myeloma", "blood cancer", "bone marrow", "car-t", "petechiae", "reed-sternberg"]):
        return ONCOLOGY_KNOWLEDGE["blood_cancer"]

    # Priority 2: Genetics & Hereditary
    if any(k in q for k in ["genetic", "hereditary", "inherited", "brca1", "brca2", "lynch syndrome", "family history", "mutation", "dna repair"]):
        return ONCOLOGY_KNOWLEDGE["genetics_hereditary"]

    # Priority 3: Treatment, side effects, chemo
    if any(k in q for k in ["side effect", "hair loss", "alopecia", "nausea", "vomiting", "neutropenia", "neuropathy", "mouth sore", "mucositis"]):
        return ONCOLOGY_KNOWLEDGE["chemotherapy_side_effects"]
    if any(k in q for k in ["chemo", "radiation", "immunotherap", "surgery", "targeted therap", "hormone therap", "how is cancer treated", "treatment", "therapy"]):
        return ONCOLOGY_KNOWLEDGE["cancer_treatments"]

    # Priority 4: Symptoms and warning signs
    if any(k in q for k in ["symptom", "warning sign", "sign", "caution", "unexplained weight loss", "how do i know", "lump", "warning"]):
        return ONCOLOGY_KNOWLEDGE["symptoms_warning_signs"]

    # Priority 5: Staging, grading, TNM, benign vs malignant
    if any(k in q for k in ["stage", "staging", "tnm", "grade", "grading", "metastasis", "metastatic", "benign", "malignant", "stage 4", "stage 0"]):
        return ONCOLOGY_KNOWLEDGE["cancer_staging"]

    # Priority 6: Prevention, risk factors, causes
    if any(k in q for k in ["prevent", "avoid", "reduce risk", "risk factor", "diet", "cause cancer", "lifestyle", "vaccine", "exercise"]):
        return ONCOLOGY_KNOWLEDGE["cancer_prevention"]

    # Priority 7: Screening, diagnosis, biopsy, scans
    if any(k in q for k in ["screen", "diagnos", "biopsy", "pet scan", "ct scan", "mri", "blood test", "tumor marker", "fna"]):
        return ONCOLOGY_KNOWLEDGE["diagnostics_screening"]

    # Priority 8: Prognosis, Survival Rates, Staging Outcomes
    if any(k in q for k in ["survival", "curable", "prognosis", "life expectancy", "can cancer be cured", "remission", "mortality"]):
        return {
            "title": "Cancer Prognosis, Remission & Survival Rates",
            "content": (
                "### 📈 Cancer Prognosis, Remission & Survival Rates\n\n"
                "**Prognosis** refers to the likely outcome or course of a disease, including the chance of recovery or recurrence.\n\n"
                "#### 📊 Understanding Key Terms:\n"
                "• **5-Year Relative Survival Rate**: The percentage of patients who live at least 5 years after diagnosis compared to the general population. Modern treatments have significantly boosted survival across many cancer types.\n"
                "• **Complete Remission (CR)**: All signs and symptoms of cancer have disappeared upon examination and diagnostic scans. When remission lasts 5+ years, many doctors consider the patient effectively cured.\n"
                "• **Partial Remission (PR)**: The tumor has shrunk significantly (usually by 50% or more), but some cancer remains detectable.\n"
                "• **Stable Disease (SD)**: The cancer is neither growing nor shrinking.\n\n"
                "#### 🩺 Critical Factors Influencing Survival:\n"
                "1. **Stage at Diagnosis**: Localized (early stage) cancers consistently have vastly superior survival rates (often >90%) compared to distant metastatic cancers.\n"
                "2. **Tumor Grade**: Low-grade (well-differentiated) tumors grow and spread much slower than high-grade aggressive tumors.\n"
                "3. **Biomarkers & Genetics**: Presence of targeted mutations (e.g., EGFR, ALK, HER2, PD-L1) enables precision therapies with high response rates.\n"
                "4. **Patient Performance Status**: Age, overall physical health, and organ function."
            )
        }

    # Priority 9: Nutrition and Diet During Cancer Treatment
    if any(k in q for k in ["nutrition", "diet", "food", "what to eat", "sugar feed cancer", "supplements"]):
        return {
            "title": "Oncology Nutrition & Dietary Guidelines",
            "content": (
                "### 🥗 Oncology Nutrition & Diet Guidelines\n\n"
                "Proper nutrition helps patients maintain strength, tolerate therapies, prevent tissue breakdown, and accelerate recovery.\n\n"
                "#### 🥦 Evidence-Based Nutrition Principles:\n"
                "• **Adequate Protein Intake**: Essential for tissue repair and immune defense (lean poultry, fish, eggs, tofu, legumes, Greek yogurt).\n"
                "• **Caloric Density**: Prevent cancer cachexia (involuntary weight and muscle loss) by consuming small, frequent nutrient-dense meals throughout the day.\n"
                "• **Plant-Forward Foods**: Colorful vegetables and berries provide natural antioxidants and phytonutrients that support cellular health.\n"
                "• **Hydration**: Aim for 2 to 3 liters of fluids daily (water, broths, electrolyte solutions), especially during chemotherapy.\n\n"
                "#### ❌ Common Myths Debunked:\n"
                "• *'Does sugar directly feed cancer?'*: All cells use glucose for fuel. While high sugar promotes obesity (a known cancer risk), eliminating all carbohydrates can cause dangerous malnutrition and weakness during chemotherapy.\n"
                "• *High-Dose Antioxidant Supplements*: Avoid unprescribed megadoses of vitamins C, E, or selenium during radiation/chemo, as they can interfere with cancer-killing oxidative mechanisms."
            )
        }

    # Priority 10: Hospital AI platform features
    if any(k in q for k in ["ai detection", "hospital system", "upload image", "model", "vit", "resnet", "efficientnet", "grad-cam"]):
        return ONCOLOGY_KNOWLEDGE["hospital_system_tools"]

    # Priority 11: General cancer definition
    if any(k in q for k in ["what is cancer", "cancer cell", "how does cancer start", "define cancer", "oncology", "tumor", "tumour"]):
        return ONCOLOGY_KNOWLEDGE["what_is_cancer"]

    return None
