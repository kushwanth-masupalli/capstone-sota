"""
cross_check.py - Clinical Report Cross-Checking and Factuality Verification Module.

Architecture Function:
  Connects the Image Classifier Output (14-finding clinical probabilities from RAD-DINO)
  with the LLM Generation ("Unchecked Report") to produce a clinically verified "Checked Report".

Features:
  1. Radiology-specific Negation & Affirmation Parser (detects "no pneumothorax", "lungs clear", etc.)
  2. Discrepancy Detection:
     - Omission: High classifier probability (>= 0.40) but report asserts absent or omits it.
     - Hallucination: Low classifier probability (< 0.15) but report asserts condition as present.
     - Concordance: Both agree on presence or absence/normalcy.
  3. Concordance Scoring (0-100% clinical alignment score).
  4. Generation of the Final "Checked Report" with verification stamp and audit addenda.
"""

import re
from typing import Dict, List, Optional, Tuple, Any

# Standard 14 CheXpert/NIH Pathologies matching project protocol
LABELS_14 = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

# Comprehensive lexical patterns for finding recognition
FINDING_PATTERNS = {
    "Cardiomegaly": [
        r"\bcardiomegaly\b",
        r"\benlarged\s+heart\b",
        r"\bcardiac\s+(?:enlargement|prominence)\b",
        r"\bheart\s+(?:is\s+)?(?:enlarged|big|large)\b",
        r"\bprominent\s+cardiac\s+silhouette\b",
        r"\bcardiac\s+silhouette\s+(?:is\s+)?(?:enlarged|prominent|big)\b",
        r"\bct\s+ratio\b",  # cardiothoracic ratio mention
        r"\bheart\s+size\s+(?:is\s+)?(?:enlarged|increased|big|large|upper\s+limits)\b",
        r"\bmildly\s+enlarged\s+heart\b",
    ],
    "Enlarged Cardiomediastinum": [
        r"\benlarged\s+cardiomediastinum\b",
        r"\bmediastin(?:al|um)\s+(?:widening|prominence|enlarged|widened|is\s+)?(?:widened|enlarged)\b",
        r"\bwidening\s+of\s+the\s+mediastinum\b",
        r"\bprominent\s+superior\s+mediastinum\b",
        r"\bmediastinal\s+contour\s+(?:is\s+)?(?:widened|enlarged|abnormal)\b",
        r"\bwidened\s+mediastinum\b",
    ],
    "Lung Opacity": [
        r"\bopacit(?:y|ies)\b",
        r"\binfiltrat(?:e|es|ion)\b",
        r"\bhaziness\b",
        r"\bairspace\s+(?:disease|shadowing|process|opacity)\b",
        r"\bpatch(?:y)?\s+opacit(?:y|ies)\b",
        r"\bopacity\s+(?:in|of|projecting)\b",
        r"\b opacities\b",
        r"\blinear\s+opacity\b",
        r"\bpulmonary\s+opacity\b",
    ],
    "Lung Lesion": [
        r"\b(?:lung\s+)?nodule(?:s)?\b",
        r"\b(?:lung\s+)?mass(?:es)?\b",
        r"\bgranuloma(?:ta|s)?\b",
        r"\blesion(?:s)?\b",
        r"\bcalcified\s+(?:granuloma|lymph)\b",
        r"\bpulmonary\s+ nodule\b",
        r"\brecent\b",
    ],
    "Edema": [
        r"\b(?:pulmonary\s+)?edema\b",
        r"\bvascular\s+congestion\b",
        r"\bcephalization\b",
        r"\binterstitial\s+(?:markings|prominence|edema)\b",
        r"\b pulmonary\s+ edema\b",
        r"\bcongestion\s+(?:of\s+)?(?:the\s+)?(?:lung|pulmonary)\b",
    ],
    "Consolidation": [
        r"\bconsolidat(?:ion|ions|ing)\b",
        r"\bairspace\s+consolidation\b",
        r"\bfocal\s+consolidation\b",
        r"\balveolar\s+consolidation\b",
        r"\b consolidation\b",
        r"\bareas\s+of\s+consolidation\b",
    ],
    "Pneumonia": [
        r"\bpneumonia\b",
        r"\binfectious\s+process\b",
        r"\bbronchopneumonia\b",
        r"\bfindings\s+(?:are|is|consistent|suspicious)\s+(?:for|of|with)\s+(?:an\s+)?pneumonia\b",
        r"\bpneumonic\b",
        r"\b infectious\s+(?:process|infiltrate|change)\b",
    ],
    "Atelectasis": [
        r"\batelecta(?:sis|tic)\b",
        r"\bvolume\s+loss\b",
        r"\bplate-?like\s+opacit(?:y|ies)\b",
        r"\bbasilar\s+collapse\b",
        r"\blinear\s+opacit(?:y|ies)\b",
        r"\b subsegmental\s+atelectasis\b",
        r"\b minor\s+atelectasis\b",
        r"\b bibasilar\s+atelectasis\b",
        r"\b mild\s+atelectasis\b",
    ],
    "Pneumothorax": [
        r"\bpneumothorax\b",
        r"\bapical\s+pneumothorax\b",
        r"\bpleural\s+air\b",
        r"\bpneumothoraces\b",
        r"\bpneumothorax\s+is\s+(?:not\s+)?(?:seen|identified|present)\b",
    ],
    "Pleural Effusion": [
        r"\b(?:pleural\s+)?effusion(?:s)?\b",
        r"\bpleural\s+fluid\b",
        r"\bblunting\s+of\s+(?:the\s+)?(?:costophrenic|cp)\s+angle(?:s)?\b",
        r"\bcostophrenic\s+angle\s+blunting\b",
        r"\bsmall\s+(?:to\s+)?(?:moderate\s+)?(?:pleural\s+)?effusion\b",
        r"\bbilateral\s+(?:pleural\s+)?effusion\b",
        r"\bfluid\s+(?:in\s+)?(?:the\s+)?(?:pleural|right\s+)?(?:space|angle)\b",
        r"\bcostophrenic\s+angles\s+(?:are\s+)?(?:blunted|obtuse)\b",
    ],
    "Pleural Other": [
        r"\bpleural\s+(?:thickening|calcification|plaque|scarring)\b",
        r"\bapical\s+capping\b",
        r"\bfibrothorax\b",
        r"\b pleural\s+(?:thickening|scarring|plaque)\b",
        r"\bpleural\s+parenchymal\s+scarring\b",
    ],
    "Fracture": [
        r"\bfracture(?:s|d)?\b",
        r"\bbroken\s+bone\b",
        r"\bcallus\s+formation\b",
        r"\bdeformity\s+of\s+(?:the\s+)?(?:rib|spine|vertebra|clavicle|bone)\b",
        r"\b rib\s+ fracture\b",
        r"\b healed\s+rib\s+ fracture\b",
        r"\b vertebral\s+ fracture\b",
        r"\b bony\s+ injury\b",
        r"\b osseous\s+ injury\b",
    ],
    "Support Devices": [
        r"\bpacemaker\b",
        r"\bcatheter\b",
        r"\bpicc\b",
        r"\bendotracheal\s+tube\b",
        r"\bchest\s+tube\b",
        r"\bsternotomy\s+(?:wires)?\b",
        r"\bhardware\b",
        r"\bstent\b",
        r"\bprosthes(?:is|es)\b",
        r"\bvascular\s+line\b",
        r"\bsurgical\s+ clips\b",
        r"\bcabg\s+ markers\b",
        r"\b medical\s+ appendage\b",
        r"\b intravascular\s+ device\b",
        r"\b central\s+ line\b",
        r"\b port\b",
    ],
    "Lung Opacity": [
        r"\bopacit(?:y|ies)\b",
        r"\binfiltrat(?:e|es|ion)\b",
        r"\bhaziness\b",
        r"\bairspace\s+(?:disease|shadowing|process|opacity)\b",
        r"\bpatch(?:y)?\s+opacit(?:y|ies)\b",
        r"\bopacity\s+(?:in|of|projecting)\b",
        r"\b opacities\b",
        r"\blinear\s+opacity\b",
        r"\bpulmonary\s+opacity\b",
    ],
}

# General anatomy normalcy indicators in radiograph reports
ANATOMY_NORMAL_PATTERNS = {
    "lungs_clear": [
        r"\blungs\s+(?:are\s+)?clear\b",
        r"\bboth\s+lungs\s+are\s+clear\b",
        r"\blungs\s+are\s+well-?expanded\s+and\s+clear\b",
        r"\bclear\s+lungs\b",
        r"\bfree\s+of\s+(?:focal\s+)?airspace\s+disease\b",
        r"\blungs\s+appear\s+(?:clear|unremarkable)\b",
        r"\bno\s+(?:focal\s+)?airspace\s+(?:disease|opacity|consolidation|infiltrate)\b",
        r"\bno\s+focal\s+(?:airspace\s+)?(?:disease|process|infiltrate|opacity)\b",
        r"\bthe\s+lungs\s+are\s+(?:clear|bilaterally\s+clear)\b",
        r"\blungs\s+are\s+(?:clear|without\s+)?(?:opacity|infiltrate|consolidation|edema)\b",
        r"\bclear\s+without\s+(?:acute\s+)?(?:findings|abnormality|airspace\s+disease)\b",
    ],
    "heart_normal": [
        r"\bheart\s+(?:size\s+)?(?:is\s+)?normal\b",
        r"\bnormal\s+heart\s+size\b",
        r"\bcardiac\s+silhouette\s+(?:is\s+)?(?:normal|within\s+normal\s+limits|unremarkable)\b",
        r"\bcardiomediastinal\s+silhouette\s+(?:is\s+)?(?:normal|within\s+normal\s+limits|unremarkable)\b",
        r"\bcardiac\s+and\s+mediastinal\s+silhouettes?\s+(?:are\s+)?(?:within\s+)?normal\b",
        r"\bthe\s+heart\s+is\s+(?:normal|unremarkable)\s+in\s+(?:size|appearance)\b",
        r"\bsilhouette\s+(?:is|are)\s+(?:within\s+)?normal\s+limits\b",
        r"\bcardiomediastinal\s+contours?\s+(?:are|is)\s+(?:unremarkable|normal|within\s+normal)\b",
        r"\bthe\s+heart\s+size\s+is\s+(?:within\s+)?(?:normal|unremarkable)\b",
        r"\bnormal\s+cardiac\s+silhouette\b",
    ],
    "bones_normal": [
        r"\b(?:osseous|skeletal|bony)\s+structures?\s+(?:are\s+)?(?:intact|normal|unremarkable)\b",
        r"\bno\s+acute\s+bony\s+(?:findings|abnormalities|abnormality)\b",
        r"\bno\s+fracture\b",
        r"\bno\s+acute\s+ossecous\s+abnormality\b",
        r"\b osseous\s+ structures\s+ (?:are|appear)\s+ (?:intact|unremarkable|normal)\b",
        r"\bthe\s+(?:thoracic\s+)?(?:skeletal|bony|osseous)\s+structures\s+(?:are|appear)\s+(?:intact|unremarkable)\b",
        r"\b no\s+ acute\s+ osseous\s+ abnormality\b",
        r"\bbony\s+thorax\s+(?:is|are)\s+(?:unremarkable|intact)\b",
        r"\bvisualized\s+bony\s+structures\s+are\s+(?:grossly\s+)?intact\b",
    ],
    "pleura_normal": [
        r"\bno\s+pleural\s+effusion\b",
        r"\bno\s+effusion\b",
        r"\bcostophrenic\s+angles\s+are\s+sharp\b",
        r"\bno\s+pleural\s+effusion\s+or\s+pneumothorax\b",
        r"\bpleural\s+spaces\s+are\s+(?:clear|unremarkable)\b",
        r"\bno\s+effusion\s+or\s+pneumothorax\b",
        r"\bcostophrenic\s+angles\s+(?:remain|are)\s+sharp\b",
        r"\bno\s+large\s+pleural\s+effusion\b",
    ],
    "mediastinum_normal": [
        r"\bmediastin(?:al|um)\s+(?:contours?|silhouette)?\s*(?:is|are)?\s*(?:within\s+)?normal\b",
        r"\bthe\s+mediastinum\s+is\s+(?:unremarkable|normal)\b",
        r"\bmediastinal\s+contours?\s+are\s+(?:unremarkable|within\s+normal\s+limits)\b",
        r"\btraaachea\s+is\s+(?:midline|central|unremarkable)\b",
        r"\bmediastinal\s+and\s+hilar\s+contours?\s+are\s+(?:unremarkable|normal|within\s+normal\s+limits)\b",
    ],
    "aorta_normal": [
        r"\baorta\s+(?:is|are)\s+(?:unremarkable|normal|tortuous\s+but\s+stable)\b",
        r"\bno\s+aortic\s+(?:aneurysm|dilation|ultrasound)\b",
        r"\bno\s+aortic\s+catarств\b",
        r"\bno\s+aortic\s+pathology\b",
    ],
    "pulmonary_vessels_normal": [
        r"\bpulmonary\s+vascularity\s+(?:is|are)\s+(?:within\s+normal\s+limits|unremarkable|normal)\b",
        r"\bno\s+pulmonary\s+vascular\s+congestion\b",
        r"\bnormal\s+pulmonary\s+vessels\b",
        r"\bno\s+evidence\s+of\s+(?:pulmonary\s+)?(?:edema|congestion)\b",
    ],
}

# Negation scope prefixes
NEGATION_PREFIXES = [
    r"\bno\s+",
    r"\bwithout\s+",
    r"\bnegative\s+for\s+",
    r"\bfree\s+of\s+",
    r"\bno\s+evidence\s+of\s+",
    r"\bno\s+sign\s+of\s+",
    r"\bno\s+definite\s+",
    r"\bno\s+focal\s+",
    r"\bno\s+acute\s+",
    r"\bno\s+visible\s+",
    r"\brules?\s+out\s+",
    r"\babsence\s+of\s+",
    r"\bunremarkable\s+for\s+",
    r"\bcleared\s+",
    r"\bnormal\s+limits\s+without\s+",
    r"\bnot\s+seen\b",
]


class ClinicalCrossChecker:
    """
    Cross-checks an unchecked LLM radiology report against the visual classifier
    pathology probabilities to detect omissions, verify concordances, and flag hallucinations.
    """

    def __init__(
        self,
        high_risk_threshold: float = 0.40,
        low_risk_threshold: float = 0.15,
        critical_alert_threshold: float = 0.50,
    ):
        self.high_risk_threshold = high_risk_threshold
        self.low_risk_threshold = low_risk_threshold
        self.critical_alert_threshold = critical_alert_threshold

    def parse_findings_in_text(self, report_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Parses report text and determines the presence status (PRESENT, ABSENT, UNMENTIONED)
        of each of the 14 conditions, taking into account medical negation and anatomy normalcy.
        """
        text = " " + str(report_text).lower().strip() + " "
        sentences = [s.strip() for s in re.split(r"[.;\n]", text) if s.strip()]

        # General anatomical flags
        lungs_clear = any(re.search(pat, text) for pat in ANATOMY_NORMAL_PATTERNS["lungs_clear"])
        heart_normal = any(re.search(pat, text) for pat in ANATOMY_NORMAL_PATTERNS["heart_normal"])
        bones_normal = any(re.search(pat, text) for pat in ANATOMY_NORMAL_PATTERNS["bones_normal"])
        mediastinum_normal = any(re.search(pat, text) for pat in ANATOMY_NORMAL_PATTERNS["mediastinum_normal"])

        results = {}

        for pathology in LABELS_14:
            if pathology == "No Finding":
                # Evaluated separately based on overall report
                continue

            patterns = FINDING_PATTERNS.get(pathology, [])
            status = "UNMENTIONED"
            matched_sentence = ""

            # Check direct mentions in sentences
            for sentence in sentences:
                for pat in patterns:
                    match = re.search(pat, sentence)
                    if match:
                        matched_sentence = sentence
                        # Check if negated in this sentence
                        prefix = sentence[:match.start()]
                        is_negated = any(re.search(neg, prefix) for neg in NEGATION_PREFIXES)
                        # Also check post-negation (e.g., 'pleural effusion is not seen')
                        suffix = sentence[match.end():]
                        if re.search(r"\b(?:is\s+)?not\s+(?:seen|identified|present|evident)\b", suffix):
                            is_negated = True

                        if is_negated:
                            status = "ABSENT"
                        else:
                            status = "PRESENT"
                        break
                if status == "PRESENT":
                    break

            # Fallback to general anatomical normal statements if unmentioned
            if status == "UNMENTIONED":
                if pathology in ["Cardiomegaly"] and heart_normal:
                    status = "ABSENT"
                    matched_sentence = "Cardiomediastinal silhouette normal"
                elif pathology in ["Enlarged Cardiomediastinum"] and (heart_normal or mediastinum_normal):
                    status = "ABSENT"
                    matched_sentence = "Mediastinal silhouette normal"
                elif pathology in ["Consolidation", "Pneumonia", "Lung Opacity", "Edema"] and lungs_clear:
                    status = "ABSENT"
                    matched_sentence = "Lungs are clear"
                elif pathology in ["Fracture"] and bones_normal:
                    status = "ABSENT"
                    matched_sentence = "Osseous structures intact"

            results[pathology] = {
                "status": status,
                "matched_sentence": matched_sentence,
            }

        # Determine 'No Finding' status
        any_present = any(res["status"] == "PRESENT" for res in results.values())
        results["No Finding"] = {
            "status": "PRESENT" if not any_present else "ABSENT",
            "matched_sentence": "No acute findings reported" if not any_present else "Abnormal findings identified",
        }

        return results

    def cross_check(
        self,
        report_text: str,
        probabilities: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Executes full cross-check between report text and classifier output.
        """
        text_findings = self.parse_findings_in_text(report_text)

        audit_matrix = []
        concordance_points = 0
        total_evaluable = 0

        omissions = []
        hallucinations = []
        concordances = []

        for pathology in LABELS_14:
            if pathology not in probabilities:
                continue

            prob = float(probabilities[pathology])
            text_info = text_findings.get(pathology, {"status": "UNMENTIONED", "matched_sentence": ""})
            text_status = text_info["status"]
            snippet = text_info["matched_sentence"]

            eval_category = "CONCORDANT_NORMAL"
            badge = "VERIFIED"
            note = ""

            if pathology == "No Finding":
                # Special evaluation for overall normality
                if prob >= 0.50 and text_status == "PRESENT":
                    eval_category = "CONCORDANT_NORMAL"
                    badge = "CONCORDANT"
                    note = f"Both classifier ({prob:.1%}) and report agree case is normal."
                    concordance_points += 1
                elif prob < 0.30 and text_status == "PRESENT":
                    eval_category = "POTENTIAL_HALLUCINATION"
                    badge = "CAUTION"
                    note = f"Classifier indicates abnormal pathology (Normal prob only {prob:.1%}), but report claims clear."
                else:
                    eval_category = "NEUTRAL"
                    badge = "NEUTRAL"
                    note = f"Normal probability: {prob:.1%}."
                    concordance_points += 0.5
                total_evaluable += 1

            else:
                total_evaluable += 1
                if prob >= self.high_risk_threshold:
                    if text_status == "PRESENT":
                        eval_category = "CONCORDANT_CONFIRMED"
                        badge = "CONFIRMED"
                        note = f"Strong agreement: Classifier ({prob:.1%}) and report both confirm finding."
                        concordance_points += 1
                        concordances.append((pathology, prob))
                    elif text_status == "ABSENT":
                        eval_category = "CRITICAL_OMISSION"
                        badge = "OMISSION"
                        note = f"DISCREPANCY: Classifier detected high risk ({prob:.1%}), but report explicitly stated absent/normal."
                        omissions.append((pathology, prob, "denied"))
                    else:  # UNMENTIONED
                        eval_category = "CRITICAL_OMISSION"
                        badge = "OMISSION"
                        note = f"OMISSION: Classifier detected high risk ({prob:.1%}), but report failed to describe it."
                        omissions.append((pathology, prob, "unmentioned"))

                elif prob <= self.low_risk_threshold:
                    if text_status == "PRESENT":
                        eval_category = "POTENTIAL_HALLUCINATION"
                        badge = "HALLUCINATION"
                        note = f"POTENTIAL HALLUCINATION: Report asserts finding is present, but classifier risk is very low ({prob:.1%})."
                        hallucinations.append((pathology, prob))
                    else:
                        eval_category = "CONCORDANT_NORMAL"
                        badge = "VERIFIED"
                        note = f"Concordant normal: Classifier confirms low risk ({prob:.1%}) and report concurs."
                        concordance_points += 1
                        concordances.append((pathology, prob))

                else:
                    # Intermediate range
                    if text_status == "PRESENT":
                        eval_category = "INDETERMINATE_POSITIVE"
                        badge = "BORDERLINE"
                        note = f"Report identified finding with moderate classifier confidence ({prob:.1%})."
                        concordance_points += 0.8
                    elif text_status == "ABSENT":
                        eval_category = "INDETERMINATE_NEGATIVE"
                        badge = "BORDERLINE"
                        note = f"Report cleared finding with moderate classifier confidence ({prob:.1%})."
                        concordance_points += 0.8
                    else:
                        eval_category = "INDETERMINATE"
                        badge = "UNMENTIONED"
                        note = f"Moderate classifier risk ({prob:.1%}) unmentioned in text."
                        concordance_points += 0.5

            audit_matrix.append({
                "pathology": pathology,
                "probability": prob,
                "text_status": text_status,
                "evaluation": eval_category,
                "badge": badge,
                "snippet": snippet,
                "clinical_note": note,
            })

        # Sort matrix: Omissions first, then Hallucinations, then confirmed positives, then normals
        order_map = {
            "CRITICAL_OMISSION": 0,
            "POTENTIAL_HALLUCINATION": 1,
            "CONCORDANT_CONFIRMED": 2,
            "INDETERMINATE_POSITIVE": 3,
            "INDETERMINATE": 4,
            "INDETERMINATE_NEGATIVE": 5,
            "CONCORDANT_NORMAL": 6,
            "NEUTRAL": 7,
        }
        audit_matrix.sort(key=lambda x: (order_map.get(x["evaluation"], 99), -x["probability"]))

        # Calculate Clinical Concordance Score
        concordance_score = (concordance_points / max(total_evaluable, 1)) * 100.0

        # Construct Checked Report
        checked_report, verification_status, addendum_text = self._build_checked_report(
            original_report=report_text,
            omissions=omissions,
            hallucinations=hallucinations,
            concordance_score=concordance_score,
        )

        return {
            "original_report": report_text,
            "checked_report": checked_report,
            "verification_status": verification_status,
            "concordance_score": round(concordance_score, 1),
            "audit_matrix": audit_matrix,
            "omissions_count": len(omissions),
            "hallucinations_count": len(hallucinations),
            "concordant_count": len(concordances),
            "addendum_text": addendum_text,
        }

    def _build_checked_report(
        self,
        original_report: str,
        omissions: List[Tuple[str, float, str]],
        hallucinations: List[Tuple[str, float]],
        concordance_score: float,
    ) -> Tuple[str, str, str]:
        """
        Builds the finalized 'Checked Report' text incorporating verification stamps,
        warning annotations, and clinical addenda.
        """
        clean_text = original_report.strip()
        addenda = []

        # Determine verification badge
        if len(omissions) == 0 and len(hallucinations) == 0:
            status = "VERIFIED_CONCORDANT"
            stamp = f" [CROSS-CHECK AUDIT: VERIFIED CONCORDANT ({concordance_score:.1f}% ALIGNMENT)]"
        elif len(omissions) > 0:
            status = "FLAGGED_OMISSION"
            stamp = f" [CROSS-CHECK AUDIT: FLAGGED DISCREPANCY ({concordance_score:.1f}% ALIGNMENT)]"
            for item in omissions:
                pathology, prob, kind = item
                addenda.append(
                    f"CLINICAL ADVISORY: Classifier visual features indicate elevated probability of {pathology.upper()} ({prob:.1%}), "
                    f"which was {kind} in draft report. Re-inspection recommended."
                )
        else:
            status = "FLAGGED_HALLUCINATION"
            stamp = f" [CROSS-CHECK AUDIT: UNCORROBORATED FINDING ({concordance_score:.1f}% ALIGNMENT)]"
            for item in hallucinations:
                pathology, prob = item
                addenda.append(
                    f"CLINICAL ADVISORY: Finding '{pathology}' asserted in report has low visual support ({prob:.1%}). Correlate with clinical history."
                )

        addendum_full = "\n".join(addenda)
        if addenda:
            final_report = f"{clean_text}\n\n--- CLINICAL CROSS-CHECK ADDENDUM ---\n" + addendum_full
        else:
            final_report = clean_text

        return final_report, status, addendum_full


AFFIRMATIVE_STATEMENTS = {
    "Cardiomegaly": "cardiomegaly is present with enlargement of the cardiac silhouette",
    "Enlarged Cardiomediastinum": "there is widening of the superior mediastinum",
    "Pleural Effusion": "pleural effusion is identified with blunting of the costophrenic angles",
    "Pneumothorax": "there is evidence of pneumothorax",
    "Consolidation": "focal airspace consolidation is identified",
    "Edema": "mild pulmonary vascular congestion and interstitial edema noted",
    "Atelectasis": "subsegmental atelectasis noted at the lung bases",
    "Lung Opacity": "patchy opacities identified within the lung parenchyma",
    "Lung Lesion": "focal pulmonary nodule/granuloma noted",
    "Pneumonia": "findings suspicious for an infectious pneumonia process",
    "Fracture": "evidence of osseous fracture deformity",
    "Pleural Other": "pleural thickening or scarring noted",
    "Support Devices": "support apparatus in place",
}


def clean_and_refine_report(
    original_report: str,
    probabilities: Dict[str, float],
    high_threshold: float = 0.40,
    low_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Implements the explicit clinical cross-checking rule:
    1. Removes any mentioned disease from the report that is NOT substantiated by the classifier (prob < low_threshold).
    2. Adds any omitted high-confidence disease detected by the classifier (prob >= high_threshold).
    3. Produces a clean, rewritten verified clinical report without clutter.
    """
    checker = ClinicalCrossChecker(high_risk_threshold=high_threshold, low_risk_threshold=low_threshold)
    findings_analysis = checker.parse_findings_in_text(original_report)

    # 1. Detected findings in classifier (prob >= 0.25, sorted descending)
    detected_findings = [
        {"pathology": k, "probability": round(float(v) * 100, 1)}
        for k, v in sorted(probabilities.items(), key=lambda x: -x[1])
        if v >= 0.25 and k != "No Finding"
    ]
    if not detected_findings:
        no_finding_prob = probabilities.get("No Finding", 0.95)
        detected_findings = [{"pathology": "No Finding (Normal)", "probability": round(float(no_finding_prob) * 100, 1)}]

    # Split report into sentences
    raw_sentences = [s.strip() for s in re.split(r"[.;\n]", original_report) if s.strip()]

    removed_diseases = []
    omitted_diseases_added = []

    # Check for hallucinated / unsubstantiated findings (asserted PRESENT, but prob < low_threshold)
    hallucinated_pathologies = set()
    for pathology, info in findings_analysis.items():
        if pathology == "No Finding":
            continue
        prob = probabilities.get(pathology, 0.0)
        if info["status"] == "PRESENT" and prob < low_threshold:
            hallucinated_pathologies.add(pathology)
            removed_diseases.append({
                "pathology": pathology,
                "probability": round(prob * 100, 1),
                "original_snippet": info.get("matched_sentence", ""),
                "reason": f"Report claimed '{pathology}', but visual classifier found no evidence ({prob:.1%} risk). Removed.",
            })

    # Check for omitted findings (prob >= high_threshold, but status != PRESENT)
    omitted_pathologies = set()
    for pathology, prob in probabilities.items():
        if pathology == "No Finding":
            continue
        if prob >= high_threshold:
            info = findings_analysis.get(pathology, {"status": "UNMENTIONED"})
            if info["status"] != "PRESENT":
                omitted_pathologies.add(pathology)
                omitted_diseases_added.append({
                    "pathology": pathology,
                    "probability": round(prob * 100, 1),
                    "action": f"Classifier detected high risk ({prob:.1%}), omitted in draft. Added to final report.",
                })

    # Sentence-level revision
    revised_sentences = []
    replaced_normal_heart = False

    for sentence in raw_sentences:
        lower_s = sentence.lower()

        # Check if sentence contains any hallucinated pathology
        sentence_has_hallucination = False
        for hall_path in hallucinated_pathologies:
            patterns = FINDING_PATTERNS.get(hall_path, [])
            for pat in patterns:
                m = re.search(pat, lower_s)
                if m:
                    # check negation
                    prefix = lower_s[:m.start()]
                    is_neg = any(re.search(neg, prefix) for neg in NEGATION_PREFIXES)
                    if not is_neg:
                        sentence_has_hallucination = True
                        break
            if sentence_has_hallucination:
                break

        if sentence_has_hallucination:
            # Drop or correct this hallucinated sentence
            continue

        # Check if sentence claims normal heart while Cardiomegaly is omitted
        if "Cardiomegaly" in omitted_pathologies and not replaced_normal_heart:
            is_heart_sentence = any(re.search(pat, lower_s) for pat in ANATOMY_NORMAL_PATTERNS["heart_normal"])
            if is_heart_sentence:
                revised_sentences.append(AFFIRMATIVE_STATEMENTS["Cardiomegaly"])
                replaced_normal_heart = True
                continue

        revised_sentences.append(sentence)

    # If Cardiomegaly was omitted and not yet added via normal-heart replacement, add it
    if "Cardiomegaly" in omitted_pathologies and not replaced_normal_heart:
        revised_sentences.insert(0, AFFIRMATIVE_STATEMENTS["Cardiomegaly"])

    # Add other omitted findings
    for omit_path in omitted_pathologies:
        if omit_path == "Cardiomegaly":
            continue
        stmt = AFFIRMATIVE_STATEMENTS.get(omit_path, f"{omit_path.lower()} is identified")
        # Ensure it's not already in revised sentences
        if not any(omit_path.lower() in s.lower() for s in revised_sentences):
            revised_sentences.append(stmt)

    # Format the final verified report
    formatted_sentences = []
    for s in revised_sentences:
        clean = s.strip()
        if clean:
            # Capitalize first letter
            clean = clean[0].upper() + clean[1:]
            formatted_sentences.append(clean)

    final_verified_report = ". ".join(formatted_sentences)
    if final_verified_report and not final_verified_report.endswith("."):
        final_verified_report += "."

    return {
        "initial_report": original_report.strip(),
        "detected_findings": detected_findings,
        "removed_diseases": removed_diseases,
        "omitted_diseases_added": omitted_diseases_added,
        "final_verified_report": final_verified_report,
        "is_concordant": len(removed_diseases) == 0 and len(omitted_diseases_added) == 0,
    }


# Module-level convenience function
def cross_check_report(report_text: str, probabilities: Dict[str, float]) -> Dict[str, Any]:
    """Convenience wrapper for single report cross-checking."""
    checker = ClinicalCrossChecker()
    return checker.cross_check(report_text, probabilities)


if __name__ == "__main__":
    # Test case demonstrating clean_and_refine_report
    sample_text = (
        "the heart size is normal. lungs are clear without focal consolidation. "
        "there is a moderate right pneumothorax."
    )
    sample_priors = {
        "Cardiomegaly": 0.82,
        "Pneumothorax": 0.02,
        "Pleural Effusion": 0.04,
        "Consolidation": 0.03,
        "No Finding": 0.15,
    }

    result = clean_and_refine_report(sample_text, sample_priors)
    print("=== REFINED CROSS-CHECK RESULT ===")
    print("INITIAL:", result["initial_report"])
    print("DETECTED FINDINGS:", result["detected_findings"])
    print("REMOVED:", result["removed_diseases"])
    print("OMITTED ADDED:", result["omitted_diseases_added"])
    print("FINAL VERIFIED REPORT:", result["final_verified_report"])

