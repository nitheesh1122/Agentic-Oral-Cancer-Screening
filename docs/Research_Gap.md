# Research Gap

# An Agentic Vision-Language Framework for Explainable Oral Cancer Screening using Evidence Verification and Confidence-Aware Clinical Reasoning

---

# Background

Artificial Intelligence has significantly improved the automated detection and classification of oral cancer using medical images. Recent studies have explored Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), Vision-Language Models (VLMs), Explainable AI (XAI), Retrieval-Augmented Generation (RAG), and Multi-Agent Systems.

Although these approaches have demonstrated promising performance individually, they address only specific components of the diagnostic workflow. A clinically useful oral cancer screening system requires not only accurate lesion detection but also transparent reasoning, evidence-supported explanations, confidence estimation, and clinician-oriented recommendations.

The literature review reveals that no existing framework integrates all these capabilities into a single explainable diagnostic pipeline.

---

# Limitations of Existing Research

## 1. CNN-Based Approaches

CNN-based methods remain the most widely used techniques for oral cancer classification and lesion detection.

### Strengths

- High image classification accuracy
- Efficient feature extraction
- Well-established architectures
- Good performance on benchmark datasets

### Limitations

- Operate as black-box models
- Limited explainability
- Do not provide clinical reasoning
- No medical evidence verification
- Poor confidence calibration
- Sensitive to image quality variations
- Difficult to interpret predictions clinically

Although Grad-CAM and related visualization techniques improve interpretability, they do not explain *why* a prediction is medically valid.

---

## 2. Vision Transformer Approaches

Vision Transformers improve global feature learning through self-attention mechanisms.

### Strengths

- Better global contextual understanding
- Improved feature representation
- Strong performance on medical imaging tasks

### Limitations

- High computational complexity
- Large training data requirements
- Limited confidence estimation
- No evidence-supported reasoning
- No integration with medical knowledge bases
- Attention maps are not equivalent to clinical explanations

ViTs improve classification performance but still function primarily as image-based prediction models.

---

## 3. Vision-Language Models

Vision-Language Models combine visual understanding with natural language generation.

### Strengths

- Joint image-text reasoning
- Medical report generation
- Clinical question answering
- Multimodal understanding

### Limitations

- Susceptible to hallucinated responses
- Lack evidence verification
- Limited oral cancer specialization
- No confidence-aware reasoning
- Generated explanations cannot always be traced to reliable medical sources

Although VLMs improve multimodal understanding, they do not ensure trustworthy clinical recommendations.

---

## 4. Explainable AI

Explainable AI increases transparency by highlighting important image regions.

### Current Methods

- Grad-CAM
- Integrated Gradients
- Layer-wise Relevance Propagation
- Saliency Maps

### Remaining Challenges

- Visual explanations only
- No medical reasoning
- No evidence retrieval
- No confidence estimation
- Limited integration with clinical workflows

Explainability alone does not provide clinically interpretable decision support.

---

## 5. Retrieval-Augmented Generation

Medical RAG reduces hallucination by retrieving relevant knowledge before response generation.

### Strengths

- Evidence-based responses
- Improved factual accuracy
- Knowledge grounding
- Better domain adaptation

### Limitations

- Usually text-only
- No lesion localization
- No image understanding
- No explainability
- No confidence estimation

Current RAG systems are rarely integrated with medical image analysis.

---

## 6. Multi-Agent Systems

Recent studies demonstrate that multiple specialized AI agents can collaboratively solve complex reasoning tasks.

### Advantages

- Modular architecture
- Specialized decision making
- Improved reasoning quality
- Better scalability
- Easier maintenance

### Limitations

- Limited application in oral cancer diagnosis
- No Vision-Language integration
- No medical evidence verification
- No confidence-aware recommendations

Healthcare-specific agentic frameworks remain largely unexplored.

---

# Identified Research Gaps

After reviewing the existing literature, the following research gaps were identified.

### Gap 1

Existing systems primarily perform classification or detection without providing complete clinical decision support.

---

### Gap 2

Most studies focus on a single AI model rather than combining complementary techniques such as Vision-Language Models, Retrieval-Augmented Generation, Explainable AI, and Multi-Agent Systems.

---

### Gap 3

Generated clinical explanations are rarely verified using trusted medical literature.

---

### Gap 4

Current AI systems do not estimate prediction confidence before producing recommendations.

---

### Gap 5

Most explainability techniques provide only visual interpretations instead of clinically meaningful reasoning.

---

### Gap 6

No existing oral cancer screening framework integrates:

- Image Quality Assessment
- Lesion Detection
- Lesion Segmentation
- Vision-Language Understanding
- Clinical Feature Extraction
- Medical Knowledge Retrieval
- Clinical Reasoning
- Evidence Verification
- Confidence Calibration
- Recommendation Generation
- Interactive Dashboard

into one collaborative architecture.

---

# Proposed Research Solution

To address these limitations, this project proposes an **Agentic Vision-Language Framework** consisting of eleven specialized AI agents.

The proposed workflow includes:

1. Image Quality Assessment Agent
2. Lesion Detection Agent
3. Lesion Segmentation Agent
4. Vision-Language Interpretation Agent
5. Clinical Feature Extraction Agent
6. Medical RAG Agent
7. Clinical Reasoning Agent
8. Evidence Verification Agent
9. Confidence Estimation Agent
10. Recommendation Agent
11. Dashboard Agent

Each agent performs an independent task while sharing structured information with subsequent agents, enabling transparent, explainable, and evidence-supported screening.

---

# Research Questions

This research aims to answer the following questions.

1. Can an Agentic AI architecture improve the transparency of oral cancer screening?

2. Does integrating Medical RAG reduce hallucinations produced by Vision-Language Models?

3. Can evidence verification improve the trustworthiness of AI-generated clinical explanations?

4. Does confidence calibration improve the reliability of referral recommendations?

5. Can multiple specialized AI agents outperform conventional single-model diagnostic systems?

---

# Expected Contributions

The proposed research is expected to contribute:

- A novel Agentic Vision-Language framework for oral cancer screening.
- Integration of Vision-Language Models with Medical RAG.
- Evidence-supported clinical reasoning.
- Confidence-aware recommendation generation.
- Explainable AI-assisted oral lesion analysis.
- A modular framework suitable for future medical AI applications.

---

# Conclusion

Although recent advances in deep learning have significantly improved oral cancer detection, existing approaches remain limited in explainability, evidence verification, confidence estimation, and clinical reasoning.

The proposed Agentic Vision-Language Framework addresses these limitations by integrating multiple specialized AI agents into a unified diagnostic pipeline capable of producing transparent, evidence-supported, and confidence-aware recommendations for oral cancer screening.