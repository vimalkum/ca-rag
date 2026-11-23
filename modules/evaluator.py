from typing import Any, Dict, List
from modules.retrieval import Chunk, TfidfRetriever
from modules.llm_judge import ask_llm

DSA_QUESTIONS = [
  # Scope & purpose
  {"label":"Purpose_Limitation", "question":"Does the agreement clearly define the specific purposes for which data is shared and contain explicit limitations preventing use for other purposes?"},
  {"label":"Permitted_Uses", "question":"Does the agreement clearly specify permitted uses of the shared data and explicitly prohibit certain uses (e.g., marketing, profiling, resale) where relevant?"},
  {"label":"Data_Categories", "question":"Does the agreement clearly list and describe the categories of data to be shared (e.g., personal data, special category data, confidential information)?"},
  {"label":"Data_Minimization", "question":"Does the agreement include obligations to minimize data sharing to what is necessary and proportionate, such as limiting data fields or records?"},
  {"label":"Legal_Basis", "question":"Does the agreement state the lawful basis or legal grounds relied on by each party for the data sharing (e.g., consent, contract, legitimate interests, statutory obligation)?"},
  {"label":"Roles_Controller_Processor", "question":"Does the agreement clearly define the parties’ roles under data protection law (controller, joint controllers, processor) and allocate related responsibilities?"},

  # Sharing mechanics
  {"label":"Transfer_Method", "question":"Does the agreement describe how data will be transferred between parties, including transfer channels, security (e.g., encryption) and any relevant frequency or scheduling?"},
  {"label":"Data_Format_Schema", "question":"Does the agreement specify the expected data format or schema, including identifiers, metadata, or standards to be used?"},
  {"label":"Frequency_Updates", "question":"Does the agreement describe how often data will be shared or updated and any procedures for refreshes, corrections, or patches?"},
  {"label":"Data_Accuracy", "question":"Does the agreement allocate responsibilities and processes for ensuring data accuracy, validation, and correction between the parties?"},

  # Security & confidentiality
  {"label":"Security_Measures", "question":"Does the agreement describe appropriate technical and organizational security measures (e.g., encryption, access controls, logging, segregation) for the shared data?"},
  {"label":"Access_Control", "question":"Does the agreement specify who may access the shared data, under what conditions, and how access is granted, reviewed, and revoked?"},
  {"label":"Audit_Logging", "question":"Does the agreement require audit logging, monitoring, or security event logging related to access and use of the shared data?"},
  {"label":"Confidentiality", "question":"Does the agreement impose clear confidentiality obligations specific to the shared data and any related information?"},
  {"label":"Personnel_Training", "question":"Does the agreement require appropriate staff vetting, confidentiality undertakings, or data protection/security training for personnel handling the shared data?"},

  # Privacy rights & compliance
  {"label":"Data_Subject_Rights", "question":"Does the agreement explain how the parties will handle and cooperate on data subject rights requests (access, deletion, rectification, portability, objection, restriction)?"},
  {"label":"DPIA_TIA", "question":"Does the agreement address when a DPIA, TIA, or similar risk assessment is required and which party is responsible for conducting and maintaining it?"},
  {"label":"Records_of_Processing", "question":"Does the agreement require maintaining appropriate records or documentation of the data sharing and associated processing activities?"},
  {"label":"Retention_Deletion", "question":"Does the agreement define retention periods for shared data and specify procedures for deletion, anonymization, or return of data (including backups) at the end of the relationship or earlier if required?"},
  {"label":"Anonymization_Pseudonymization", "question":"Does the agreement address use of anonymization or pseudonymization and set any restrictions on re-identification of data subjects?"},
  {"label":"Transparency_Notices", "question":"Does the agreement require appropriate transparency to data subjects (e.g., privacy notices or other communications) about the data sharing, where applicable?"},

  # Breach & incident management
  {"label":"Incident_Response", "question":"Does the agreement describe coordinated incident response steps between the parties, including investigation, mitigation, and communication?"},
  {"label":"Breach_Notification", "question":"Does the agreement define personal data breach notification obligations, including triggers, timelines, and which party must notify regulators or data subjects?"},

  # Onward transfers & third parties
  {"label":"Subprocessors_Third_Parties", "question":"Does the agreement regulate the use of subprocessors or third parties, including prior approval, written contracts, and flow-down of equivalent obligations?"},
  {"label":"Onward_Disclosure", "question":"Does the agreement restrict onward disclosure or onward sharing of the data and describe any conditions or approvals required?"},

  # International transfers
  {"label":"Cross_Border_Transfers", "question":"Does the agreement address cross-border transfers of the shared data and specify the safeguards used (e.g., SCCs, adequacy decisions, binding corporate rules)?"},
  {"label":"Data_Location", "question":"Does the agreement specify where the data will be hosted, stored, or processed and any data residency requirements?"},

  # Governance & accountability
  {"label":"Point_of_Contact", "question":"Does the agreement identify points of contact or roles (e.g., DPO, security contact, escalation paths) for data protection and security matters?"},
  {"label":"Audit_Rights", "question":"Does the agreement grant rights to audit, inspect, or otherwise obtain assurance (e.g., questionnaires, certifications) regarding the other party’s compliance?"},
  {"label":"Reporting", "question":"Does the agreement require any regular reporting, metrics, attestations, or certifications to demonstrate ongoing compliance with the data sharing obligations?"},
  {"label":"Policy_Review", "question":"Does the agreement describe how often the data sharing arrangement will be reviewed or updated and how changes will be notified to the parties?"},

  # Liability & enforcement
  {"label":"Indemnity_Liability", "question":"Does the agreement allocate liability or indemnity for privacy or security incidents, unlawful processing, or misuse of the shared data?"},
  {"label":"Termination_Effect", "question":"Does the agreement describe termination triggers related to data protection or security and specify what happens to the shared data upon termination?"},

  # Ethics, IP, and research compliance
  {"label":"Ethical_Approval", "question":"Where relevant (e.g., research or health contexts), does the agreement address ethical approval, research governance, or oversight requirements?"},
  {"label":"Intellectual_Property", "question":"Does the agreement specify ownership and licensing of the shared data, derived datasets, models, or analyses produced using the data?"},
  {"label":"Attribution_Publication", "question":"Does the agreement set conditions for data citation, attribution, or publication of results based on the shared data?"},

  # Monitoring and compliance
  {"label":"Compliance_Assurance", "question":"Does the agreement describe mechanisms for monitoring, assurance, or periodic review of compliance with the data sharing obligations?"},
  {"label":"Sanctions_Remediation", "question":"Does the agreement describe remedies, sanctions, or remediation steps if a party fails to comply with the data sharing obligations?"}
]




def build_chunks_from_langchain_docs(docs) -> List[Chunk]:
    chunks = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        chunks.append(
            Chunk(
                text=d.page_content,
                source=str(meta.get("source", "")),
                page=meta.get("page", None),
            )
        )
    return chunks

def evaluate_document(chunks: List[Chunk], questions: List[Dict[str, str]], k: int = 6) -> Dict[str, Any]:
    retriever = TfidfRetriever(chunks)
    results = []
    answered_count = 0

    for q in questions:
        top = retriever.top_k(q["question"], k=k)
        context = "\n\n---\n\n".join([f"[p={c.page}] {c.text}" for c, _ in top])[:24000]

        out = ask_llm(q["question"], context)
        out["label"] = q["label"]
        out["question"] = q["question"]

        # Enforce types
        out["answered"] = bool(out.get("answered", False))
        out["evidence"] = out.get("evidence", []) or []
        if out["answered"]:
            answered_count += 1

        results.append(out)

    coverage = answered_count / max(1, len(questions))
    return {"coverage": coverage, "answered": answered_count, "total": len(questions), "results": results}
