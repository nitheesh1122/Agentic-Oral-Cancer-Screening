"""Command-line entry point for the research prototype."""

from __future__ import annotations

from typing import Any

from logger import configure_logging
from src.agents import (
    ClinicalFeatureAgent,
    ClinicalReasoningAgent,
    ConfidenceEstimationAgent,
    DashboardAgent,
    EvidenceVerificationAgent,
    ImageQualityAgent,
    LesionDetectionAgent,
    LesionSegmentationAgent,
    MedicalRAGAgent,
    RecommendationAgent,
    VisionLanguageAgent,
)


def _initialize_agent(agent: Any) -> None:
    """Initialize an agent defensively and surface setup failures early."""

    try:
        agent.initialize()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize {agent.__class__.__name__}.") from exc


def main() -> int:
    """Initialize the placeholder agent pipeline in sequence."""

    configure_logging()

    print("=========================================")
    print("Agentic Oral Cancer Screening Framework")
    print("=========================================")
    print()
    print("Initializing Image Quality Agent...")
    image_quality_agent = ImageQualityAgent()
    _initialize_agent(image_quality_agent)
    print("Initializing Detection Agent...")
    lesion_detection_agent = LesionDetectionAgent()
    _initialize_agent(lesion_detection_agent)
    print("Initializing Segmentation Agent...")
    lesion_segmentation_agent = LesionSegmentationAgent()
    _initialize_agent(lesion_segmentation_agent)
    print("Initializing Vision-Language Agent...")
    vision_language_agent = VisionLanguageAgent()
    _initialize_agent(vision_language_agent)
    print("Initializing Clinical Feature Agent...")
    clinical_feature_agent = ClinicalFeatureAgent()
    _initialize_agent(clinical_feature_agent)
    print("Initializing Medical RAG Agent...")
    medical_rag_agent = MedicalRAGAgent()
    _initialize_agent(medical_rag_agent)
    print("Initializing Clinical Reasoning Agent...")
    clinical_reasoning_agent = ClinicalReasoningAgent()
    _initialize_agent(clinical_reasoning_agent)
    print("Initializing Evidence Verification Agent...")
    evidence_verification_agent = EvidenceVerificationAgent()
    _initialize_agent(evidence_verification_agent)
    print("Initializing Confidence Estimation Agent...")
    confidence_estimation_agent = ConfidenceEstimationAgent()
    _initialize_agent(confidence_estimation_agent)
    print("Initializing Recommendation Agent...")
    recommendation_agent = RecommendationAgent()
    _initialize_agent(recommendation_agent)
    print("Initializing Dashboard Agent...")
    dashboard_agent = DashboardAgent()
    _initialize_agent(dashboard_agent)
    print()
    print("Pipeline Initialized Successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())