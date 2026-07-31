# Literature Review

# An Agentic Vision-Language Framework for Explainable Oral Cancer Screening using Evidence Verification and Confidence-Aware Clinical Reasoning

---

## Overview

Oral cancer is one of the most prevalent cancers worldwide, particularly in developing countries. Early diagnosis significantly improves patient survival; however, conventional diagnostic procedures depend heavily on specialist expertise and invasive biopsy-based confirmation.

Recent advances in Artificial Intelligence (AI) have demonstrated promising performance in oral lesion detection and classification. Nevertheless, existing approaches often operate as black-box systems and provide limited clinical reasoning, explainability, confidence estimation, or evidence verification.

This literature review investigates recent research in:

- Oral cancer detection using Deep Learning
- Vision Transformers (ViTs)
- Vision-Language Models (VLMs)
- Explainable Artificial Intelligence (XAI)
- Retrieval-Augmented Generation (RAG)
- Multi-Agent AI Systems
- Clinical Decision Support Systems

The findings from these studies provide the foundation for the proposed agentic oral cancer screening framework.

---

# Literature Classification

The reviewed literature can be grouped into six major research areas.

| Research Area | Papers Reviewed | Purpose |
|---------------|----------------|----------|
| Oral Cancer Detection | 8 | Detection, Classification, Segmentation |
| Vision-Language Models | 2 | Multimodal medical understanding |
| Retrieval-Augmented Generation | 3 | Knowledge grounding |
| Explainable AI | 3 | Transparent diagnosis |
| Multi-Agent Systems | 3 | Agent collaboration |
| Clinical Decision Support | 1 | Clinical reasoning |

---

# Research Evolution

The evolution of AI-based oral cancer diagnosis can be summarized as:

```

CNN-based Classification
↓
Attention-based CNN
↓
Vision Transformers
↓
Vision-Language Models
↓
Explainable AI
↓
Medical RAG
↓
Multi-Agent Clinical Reasoning
↓
Proposed Agentic Framework

```

Each stage improves one aspect of diagnosis, yet no existing work integrates all components into a unified clinical decision-support system.

---

# Comparative Literature Analysis

| Category | Existing Research | Limitations | Our Improvement |
|-----------|------------------|-------------|-----------------|
| CNN | Accurate classification | Black-box predictions | Explainable diagnosis |
| Vision Transformer | Better feature learning | Image-only reasoning | VLM integration |
| Vision-Language Model | Image-text understanding | No evidence verification | Medical RAG |
| Explainable AI | Heatmaps | No clinical reasoning | Evidence-supported explanations |
| Medical RAG | Knowledge grounding | No image analysis | VLM + RAG |
| Multi-Agent AI | Collaborative reasoning | Not oral cancer specific | Agentic oral cancer framework |

---

# Summary of Reviewed Literature

## 1. Oral Cancer Detection

Multiple studies demonstrate that CNNs and Vision Transformers achieve high classification accuracy for oral lesion detection.

Common contributions include:

- Automated lesion detection
- Oral lesion classification
- Histopathology analysis
- Segmentation-assisted diagnosis

Common limitations include:

- Black-box predictions
- No explainability
- No clinical reasoning
- Limited multimodal capability

These studies provide the foundation for the Detection and Segmentation Agents.

---

## 2. Vision-Language Models

Medical Vision-Language Models integrate medical images with textual descriptions to improve diagnosis.

Reported applications include:

- Medical report generation
- Image captioning
- Clinical question answering
- Disease severity estimation

Limitations include:

- No oral cancer specialization
- No evidence verification
- No Retrieval-Augmented Generation
- Limited explainability

These studies inspire the Vision-Language Agent.

---

## 3. Retrieval-Augmented Generation

Recent RAG research demonstrates that retrieving external medical knowledge significantly reduces hallucination and improves factual correctness.

Advantages include:

- Knowledge grounding
- Medical evidence retrieval
- Improved factual responses
- Better domain adaptation

Limitations include:

- No medical image analysis
- No Vision-Language integration
- No oral cancer reasoning

These studies support the Medical RAG Agent.

---

## 4. Explainable AI

Explainable AI techniques improve clinician trust by revealing why a model reaches a prediction.

Common techniques include:

- Grad-CAM
- Integrated Gradients
- Layer-wise Relevance Propagation
- Occlusion Analysis
- Saliency Maps

Current limitations include:

- Mostly visualization-based explanations
- Lack of clinical reasoning
- Limited multimodal support

These works support the Explainability and Evidence Verification Agents.

---

## 5. Multi-Agent Systems

Recent LLM-based Multi-Agent Systems distribute complex reasoning across specialized agents.

Typical agent roles include:

- Planner
- Reviewer
- Reasoner
- Verifier
- Evaluator

Advantages:

- Modular architecture
- Improved reasoning
- Better collaboration
- Reduced hallucination

Limitations:

- Not designed for oral cancer diagnosis
- No medical imaging integration
- No confidence estimation

These studies form the architectural foundation of the proposed framework.

---

## 6. Clinical Decision Support

Clinical decision-support systems assist healthcare professionals by combining diagnostic predictions with medical knowledge.

Current systems typically provide:

- Diagnostic assistance
- Treatment recommendations
- Clinical workflow support

However, they generally lack:

- Vision-Language understanding
- Explainability
- Confidence-aware reasoning
- Multi-agent collaboration

These observations motivate the proposed Clinical Reasoning Agent.

---

# Research Gaps

Based on the reviewed literature, several important research gaps remain.

1. Existing studies focus mainly on image classification and detection.

2. Very few integrate Vision-Language Models for oral lesion understanding.

3. Current systems do not verify explanations using retrieved medical evidence.

4. Confidence estimation is rarely incorporated into diagnostic pipelines.

5. Most explainable AI techniques provide only visual explanations without clinical reasoning.

6. Multi-Agent AI has not been effectively applied to oral cancer diagnosis.

7. No unified framework combines:

- Oral lesion detection
- Segmentation
- Vision-Language Models
- Medical Retrieval-Augmented Generation
- Explainable AI
- Evidence Verification
- Confidence Estimation
- Clinical Reasoning
- Multi-Agent Collaboration

---

# Proposed Research Direction

To address these limitations, this project proposes an integrated Agentic Vision-Language Framework consisting of:

- Image Quality Assessment Agent
- Lesion Detection Agent
- Lesion Segmentation Agent
- Vision-Language Agent
- Clinical Feature Extraction Agent
- Medical RAG Agent
- Clinical Reasoning Agent
- Evidence Verification Agent
- Confidence Estimation Agent
- Recommendation Agent
- Dashboard Agent

Each agent performs a specialized task while collaboratively producing an explainable and evidence-supported clinical diagnosis.

---

# Conclusion

The reviewed literature demonstrates substantial progress in oral cancer diagnosis, Vision-Language Models, Explainable AI, Retrieval-Augmented Generation, and Multi-Agent Systems. However, these technologies have largely been developed independently.

No existing study integrates multimodal lesion analysis, medical knowledge retrieval, explainable reasoning, evidence verification, confidence estimation, and collaborative AI agents into a single framework for oral cancer screening.

The proposed research aims to bridge these gaps by developing a trustworthy, explainable, and clinically interpretable Agentic Vision-Language Framework for oral cancer screening.

---

# References

The detailed analysis of each reviewed paper is available in:

- `papers/`
- `docs/literature_survey.pdf`

This document serves as the consolidated literature review for the proposed framework.