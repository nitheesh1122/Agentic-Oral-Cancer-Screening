# Project Proposal

# An Agentic Vision-Language Framework for Explainable Oral Cancer Screening using Evidence Verification and Confidence-Aware Clinical Reasoning

---

## 1. Project Overview

Oral cancer is one of the leading causes of cancer-related mortality worldwide. Despite significant advances in medical imaging and artificial intelligence, many patients are still diagnosed at advanced stages due to delayed detection, limited access to specialists, and the absence of reliable screening tools in low-resource settings.

Recent deep learning approaches have demonstrated encouraging performance in oral lesion classification and detection. However, most existing systems function as black-box models that provide predictions without explaining the underlying reasoning, verifying supporting evidence, or communicating the confidence of their decisions. These limitations reduce clinician trust and restrict the practical adoption of AI-assisted diagnosis.

This project proposes an **Agentic Vision-Language Framework** that combines computer vision, multimodal reasoning, medical knowledge retrieval, evidence verification, and confidence-aware clinical reasoning into a unified decision-support system for oral cancer screening.

The proposed framework is intended to assist clinicians by providing transparent, evidence-supported, and confidence-aware recommendations rather than replacing professional medical judgment.

---

# 2. Problem Statement

Current AI-based oral cancer screening systems primarily focus on image classification or lesion detection. Although these approaches often achieve high predictive performance, they exhibit several important limitations:

- Lack of explainability for clinical predictions.
- Limited understanding of visual and textual medical information together.
- No verification of generated clinical explanations using trusted medical evidence.
- Absence of confidence estimation for uncertain predictions.
- Minimal support for structured clinical reasoning.
- Lack of collaboration among specialized AI components.

These limitations reduce the reliability and interpretability of existing systems in real-world clinical environments.

---

# 3. Motivation

The motivation behind this project is to develop a trustworthy AI-assisted screening framework capable of producing clinically meaningful outputs instead of isolated classification results.

The proposed system aims to:

- Improve transparency through explainable AI.
- Reduce hallucinated medical explanations using evidence retrieval.
- Support clinicians with structured reasoning.
- Estimate prediction confidence before generating recommendations.
- Combine multiple specialized AI agents into a collaborative diagnostic workflow.

---

# 4. Objectives

The primary objectives of this research are:

- Develop an agent-based framework for oral cancer screening.
- Assess image quality before analysis.
- Detect suspicious oral lesions automatically.
- Segment lesion regions for precise localization.
- Interpret lesion characteristics using a Vision-Language Model.
- Extract clinically relevant lesion features.
- Retrieve supporting medical evidence using Retrieval-Augmented Generation.
- Verify generated clinical claims against retrieved evidence.
- Estimate prediction confidence using calibration techniques.
- Generate clinician-friendly recommendations.
- Compare the proposed framework with conventional CNN- and Vision Transformer-based methods.

---

# 5. Proposed Framework

The proposed framework consists of eleven specialized AI agents working collaboratively throughout the diagnostic workflow.

### Agent Workflow

1. Image Quality Assessment Agent
2. Lesion Detection Agent
3. Lesion Segmentation Agent
4. Vision-Language Interpretation Agent
5. Clinical Feature Extraction Agent
6. Medical Retrieval Agent
7. Clinical Reasoning Agent
8. Evidence Verification Agent
9. Confidence Estimation Agent
10. Recommendation Agent
11. Dashboard Agent

Each agent performs an independent task while sharing structured information with subsequent agents to produce an explainable and evidence-supported screening report.

---

# 6. Proposed Technologies

| Component | Selected Technology |
|-----------|---------------------|
| Image Quality Assessment | BRISQUE + OpenCV |
| Lesion Detection | YOLOv11 |
| Lesion Segmentation | SAM 2 |
| Vision-Language Understanding | MedGemma |
| Medical Knowledge Retrieval | FAISS + LangChain |
| Agent Orchestration | LangGraph |
| Confidence Calibration | Platt Scaling |
| User Interface | Streamlit |
| Backend API | FastAPI |

---

# 7. Research Methodology

The project will be carried out in the following stages.

### Phase 1 – Literature Review

- Review recent research on oral cancer diagnosis.
- Study Vision Transformers and Vision-Language Models.
- Analyze Explainable AI methods.
- Study Retrieval-Augmented Generation.
- Investigate Multi-Agent AI systems.
- Identify research gaps.

### Phase 2 – Dataset Preparation

- Collect oral lesion image datasets.
- Verify licensing and usage permissions.
- Organize datasets into training, validation, and testing sets.
- Perform preprocessing and annotation validation.

### Phase 3 – AI Model Development

- Implement image quality assessment.
- Train lesion detection models.
- Develop lesion segmentation.
- Integrate Vision-Language reasoning.

### Phase 4 – Knowledge Grounding

- Build a medical knowledge base.
- Create vector embeddings.
- Implement FAISS indexing.
- Retrieve relevant clinical evidence.

### Phase 5 – Clinical Reasoning

- Coordinate specialized AI agents.
- Verify generated clinical claims.
- Estimate confidence scores.
- Generate explainable recommendations.

### Phase 6 – Evaluation

Evaluate the framework using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- mAP
- Dice Score
- IoU
- Expected Calibration Error (ECE)
- Brier Score

The proposed framework will also be compared against CNN and Vision Transformer baselines.

---

# 8. Expected Outcomes

The expected deliverables of this project include:

- An explainable AI-assisted oral cancer screening framework.
- A modular multi-agent architecture.
- Automated lesion localization and segmentation.
- Vision-Language-based clinical interpretation.
- Evidence-supported diagnostic explanations.
- Confidence-aware clinical recommendations.
- An interactive web dashboard.
- Comparative experimental evaluation with existing methods.

---

# 9. Expected Contributions

This research is expected to contribute:

- An integrated Agentic AI framework for oral cancer screening.
- Combination of Vision-Language Models with Medical RAG.
- Evidence verification before recommendation generation.
- Confidence-aware clinical decision support.
- Improved transparency and interpretability of AI-assisted diagnosis.
- A modular architecture that can be extended for other medical imaging applications.

---

# 10. Project Scope

### In Scope

- Oral lesion image analysis.
- AI-assisted screening support.
- Explainable decision-making.
- Medical evidence retrieval.
- Confidence estimation.
- Research prototype development.

### Out of Scope

- Clinical diagnosis.
- Treatment recommendation.
- Replacement of biopsy or histopathological examination.
- Autonomous medical decision-making.
- Clinical deployment without regulatory approval.

---

# 11. Project Status

The project has successfully completed:

- Problem identification
- Literature survey
- Research gap analysis
- Proposed framework design
- Technology selection
- Repository setup
- Initial software architecture

The next phase focuses on dataset preparation, AI model implementation, agent integration, experimental evaluation, and dashboard development.

---

# 12. Conclusion

The proposed Agentic Vision-Language Framework aims to address the limitations of existing oral cancer screening systems by integrating computer vision, multimodal reasoning, medical knowledge retrieval, explainable AI, evidence verification, and confidence estimation into a single collaborative framework.

By providing transparent, evidence-supported, and confidence-aware recommendations, the proposed system seeks to improve clinician trust while supporting early oral cancer screening. The framework is intended as a research prototype that complements, rather than replaces, professional clinical judgment.