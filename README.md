# 🦷 An Agentic Vision-Language Framework for Explainable Oral Cancer Screening using Evidence Verification and Confidence-Aware Clinical Reasoning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![Status](https://img.shields.io/badge/Status-Research%20%26%20Initial%20Implementation-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

# 📌 Project Overview

Oral cancer is one of the leading causes of cancer-related deaths worldwide, primarily because it is often diagnosed at advanced stages. Early diagnosis significantly improves survival rates; however, current diagnostic procedures rely heavily on biopsy and expert clinical examination, making them time-consuming, invasive, and expensive.

Recent advances in Artificial Intelligence have introduced deep learning techniques for oral lesion classification. However, most existing systems suffer from several critical limitations:

- Black-box decision making
- Lack of explainability
- No evidence verification
- No confidence estimation
- Limited clinical reasoning
- Poor trust among healthcare professionals

To address these limitations, this project proposes an **Agentic Vision-Language Framework** that combines Computer Vision, Vision-Language Models (VLMs), Retrieval-Augmented Generation (RAG), Multi-Agent AI, Evidence Verification, and Confidence Calibration to provide explainable and trustworthy oral cancer screening.

Unlike conventional CNN-based approaches, the proposed framework decomposes the diagnostic workflow into multiple specialized AI agents that collaboratively analyze oral lesion images, retrieve medical knowledge, verify generated explanations, estimate prediction confidence, and generate clinician-friendly recommendations.

---

# 🎯 Problem Statement

Current AI-based oral cancer screening systems mainly perform image classification without providing clinically explainable reasoning. These systems cannot justify why a lesion is classified as malignant, lack medical evidence retrieval, fail to verify generated explanations, and do not estimate prediction reliability.

This limits their adoption in real-world clinical environments where transparency, trust, and evidence-backed decision support are essential.

The objective of this research is to develop a modular Agentic Vision-Language Framework capable of producing explainable, evidence-grounded, and confidence-aware clinical recommendations for oral cancer screening.

---

# 🎯 Objectives

The primary objectives of this project are:

- Develop an Agentic AI framework for oral cancer screening.
- Assess oral image quality before diagnosis.
- Detect suspicious oral lesions using **YOLOv11**.
- Segment lesions accurately using **SAM 2**.
- Interpret lesion characteristics using **MedGemma Vision-Language Model**.
- Extract clinically meaningful lesion features.
- Retrieve relevant medical literature using **Medical RAG**.
- Verify generated explanations using evidence verification.
- Estimate prediction confidence using confidence calibration.
- Generate explainable clinical recommendations.
- Build an interactive dashboard for clinicians.
- Compare the proposed framework with traditional CNN and Vision Transformer approaches.

---

# 💡 Proposed Novel Contributions

This research introduces several novel components that are absent in existing oral cancer screening systems.

✅ Multi-Agent AI Architecture

✅ Vision-Language Understanding

✅ Medical Retrieval-Augmented Generation (Medical RAG)

✅ Evidence Verification Module

✅ Confidence Estimation Module

✅ Explainable Clinical Reasoning

✅ Interactive Clinical Dashboard

To the best of our literature survey, no existing oral cancer screening framework integrates all these components into a unified explainable diagnostic pipeline.

---

# 🏗️ System Architecture

```
                    Input Oral Image
                           │
                           ▼
             Image Quality Assessment Agent
                           │
                           ▼
                 Lesion Detection Agent
                     (YOLOv11)
                           │
                           ▼
              Lesion Segmentation Agent
                       (SAM 2)
                           │
                           ▼
        Vision-Language Understanding Agent
                    (MedGemma)
                           │
                           ▼
          Clinical Feature Extraction Agent
                           │
                           ▼
                Medical RAG Agent
              (FAISS + LangChain)
                           │
                           ▼
            Clinical Reasoning Agent
                   (LangGraph)
                           │
                           ▼
          Evidence Verification Agent
                           │
                           ▼
          Confidence Estimation Agent
               (Platt Scaling)
                           │
                           ▼
            Recommendation Agent
                           │
                           ▼
               Interactive Dashboard
```

---

# ⚙️ Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Object Detection | YOLOv11 |
| Segmentation | SAM 2 |
| Vision-Language Model | MedGemma |
| Image Processing | OpenCV |
| Retrieval | FAISS |
| Agent Framework | LangGraph |
| LLM Orchestration | LangChain |
| Dashboard | Streamlit |
| Backend API | FastAPI |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |

---

# 🤖 Agent Pipeline

The proposed framework consists of **11 specialized AI agents**.

| Agent | Responsibility |
|--------|----------------|
| Image Quality Assessment Agent | Blur, brightness, contrast evaluation |
| Lesion Detection Agent | Detect suspicious oral lesions |
| Lesion Segmentation Agent | Generate lesion mask |
| Vision-Language Understanding Agent | Image-language interpretation |
| Clinical Feature Extraction Agent | Extract medical lesion characteristics |
| Medical RAG Agent | Retrieve supporting medical literature |
| Clinical Reasoning Agent | Generate diagnostic reasoning |
| Evidence Verification Agent | Verify generated explanations |
| Confidence Estimation Agent | Estimate prediction reliability |
| Recommendation Agent | Generate clinician-friendly report |
| Dashboard Agent | Display all outputs interactively |

---

# 🧠 Selected Models

| Module | Model |
|--------|-------|
| Image Quality Assessment | BRISQUE + OpenCV |
| Lesion Detection | YOLOv11 |
| Lesion Segmentation | SAM 2 |
| Vision-Language Understanding | MedGemma |
| Clinical Feature Extraction | Rule-Based Analysis |
| Medical Retrieval | FAISS + LangChain |
| Clinical Reasoning | LangGraph |
| Evidence Verification | Retrieval Cross Validation |
| Confidence Estimation | Platt Scaling |
| Recommendation | Rule-Based Templates |
| Dashboard | Streamlit + FastAPI |

---

# 📂 Repository Structure

```
Oral-Cancer-Agentic-AI
│
├── assets/
│   ├── images/
│   └── diagrams/
│
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── docs/
│   ├── Proposal.pdf
│   ├── Literature_Survey.md
│   ├── Research_Gap.md
│   ├── Workflow.pdf
│   ├── Model_Selection.pdf
│   ├── Review1_PPT.pptx
│   └── PROJECT_PROGRESS.md
│
├── papers/
│   ├── Base_Paper.pdf
│   └── Literature_Papers/
│
├── notebooks/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── image_quality/
│   ├── detection/
│   ├── segmentation/
│   ├── vlm/
│   ├── clinical_features/
│   ├── rag/
│   ├── reasoning/
│   ├── verification/
│   ├── confidence/
│   ├── recommendation/
│   └── dashboard/
│
├── tests/
│
├── main.py
├── config.py
├── logger.py
├── utils.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🔄 End-to-End Workflow

1. Capture oral cavity image.
2. Assess image quality.
3. Detect suspicious lesion.
4. Segment lesion.
5. Interpret lesion using Vision-Language Model.
6. Extract clinical features.
7. Retrieve supporting medical literature.
8. Generate explainable reasoning.
9. Verify generated explanations.
10. Estimate confidence score.
11. Generate clinician recommendation.
12. Display results in an interactive dashboard.

---

# 📈 Expected Outputs

- Oral lesion localization
- Segmentation mask
- Medical image interpretation
- Clinical feature extraction
- Retrieved medical evidence
- Verified explanation
- Confidence score
- Clinical recommendation
- Downloadable PDF report
- Interactive dashboard

---

# 🚀 Current Project Progress

| Task | Status |
|------|--------|
| Literature Survey | ✅ Completed |
| Research Gap Identification | ✅ Completed |
| Problem Statement | ✅ Completed |
| Proposed Methodology | ✅ Completed |
| System Architecture | ✅ Completed |
| Model Selection | ✅ Completed |
| Repository Setup | ✅ Completed |
| Initial Code Structure | ✅ Completed |
| Dataset Collection | 🔄 In Progress |
| Initial Agent Development | 🔄 In Progress |
| Model Integration | ⏳ Planned |
| Training & Evaluation | ⏳ Planned |
| Dashboard Development | ⏳ Planned |
| Final Testing | ⏳ Planned |

---

# ⚙️ Installation

```bash
git clone https://github.com/<username>/Oral-Cancer-Agentic-AI.git

cd Oral-Cancer-Agentic-AI

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶️ Running the Project

```bash
python main.py
```

Expected Output:

```
=========================================
Agentic Oral Cancer Screening Framework
=========================================

Initializing Image Quality Agent...
Initializing Detection Agent...
Initializing Segmentation Agent...
Initializing Vision-Language Agent...
Initializing Clinical Feature Agent...
Initializing Medical RAG Agent...
Initializing Clinical Reasoning Agent...
Initializing Evidence Verification Agent...
Initializing Confidence Estimation Agent...
Initializing Recommendation Agent...
Initializing Dashboard Agent...

Pipeline Initialized Successfully.
```

---

# 📚 Research Papers

This repository includes:

- Base Research Paper
- Literature Survey Papers
- Research Gap Analysis
- Workflow Design
- Model Selection Document
- Project Proposal

The papers are organized inside the `papers/` directory and supporting documentation is available in the `docs/` folder.

---

# 👨‍💻 Team Members

| Name | Register Number |
|------|-----------------|
| Nitheesh S | 23CSR149 |
| Redhani T V | 23CSR174 |
| Sabarish K S | 23CSR184 |

---

# 🎓 Supervisor

**Ms. Nanthiya**

Assistant Professor

Department of Computer Science and Engineering

Kongu Engineering College

---

# 📅 Project Duration

July 2026 – March 2027

---

# 📄 License

This project is released under the **MIT License**.

See the `LICENSE` file for more information.

---

# ⭐ Acknowledgements

We thank the Department of Computer Science and Engineering, Kongu Engineering College, for providing the opportunity to carry out this research project.

We also acknowledge the open-source AI community for developing and maintaining frameworks such as YOLO, SAM, MedGemma, LangChain, LangGraph, FAISS, PyTorch, and Streamlit, which form the technological foundation of this work.