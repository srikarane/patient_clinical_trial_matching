from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from google.adk.agents import LlmAgent, Agent
from google.adk.tools import VertexAiSearchTool
from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Any

# Load environment variables
load_dotenv()
#DATASTORE_ID = os.getenv("VERTEX_SEARCH_DATASTORE_ID")
DATASTORE_ID =  "projects/utopian-splicer-455918-b3/locations/global/collections/default_collection/dataStores/clinical-trial-match_1746270287766"
if not DATASTORE_ID:
    raise ValueError("Missing or invalid VERTEX_SEARCH_DATASTORE_ID in .env")
MODEL_NM="gemini-2.0-flash"
# Tools
vertex_search_tool = VertexAiSearchTool(data_store_id=DATASTORE_ID)

def generate_patient_reports(validated_inclusion_data: Dict[str, Any], validated_exclusion_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates final eligibility reports based on inclusion and exclusion evaluations.

    Args:
        inclusion_data: A dictionary of inclusion evaluation results.
        exclusion_data: A dictionary of exclusion evaluation results.

    Returns:
        A dictionary containing formatted patient eligibility reports.
    """
    inclusion_map = {p['id']: p for p in validated_inclusion_data.get('patients', [])}
    exclusion_map = {p['id']: p for p in validated_exclusion_data.get('patients', [])}
    reports = []

    for patient_id, inc in inclusion_map.items():
        exc = exclusion_map.get(patient_id, {})
        matched_ex = exc.get('matched_exclusion_criteria', [])

        total = len(inc['satisfied_inclusion_criteria']) + len(inc['missing_inclusion_criteria'])
        ratio = len(inc['satisfied_inclusion_criteria']) / total if total > 0 else 0
        eligible = ratio >= 0.75 and not matched_ex

        reports.append({
            "id": patient_id,
            "satisfied_inclusion_criteria": inc.get('satisfied_inclusion_criteria', []),
            "missing_inclusion_criteria": inc.get('missing_inclusion_criteria', []),
            "matched_exclusion_criteria": matched_ex,
            "non_matched_exclusion_criteria": exc.get('non_matched_exclusion_criteria', []),
            "final_eligibility": "Eligible" if eligible else "Not Eligible"
        })

    return {"reports": reports, "status": "success"}

# Validation Agents
validate_inclusion_agent = LlmAgent(
    name="validate_inclusion_agent",
    model=MODEL_NM,
    output_key="validated_inclusion_data",
    instruction="""
Given raw inclusion criteria output from inclusion_agent, format the output into a JSON structure with a key `patients` containing a list of patient dictionaries. Each patient must include: id, satisfied_inclusion_criteria, and missing_inclusion_criteria. Only include patients who meet at least 75% of the inclusion criteria.
"""
)

validate_exclusion_agent = LlmAgent(
    name="validate_exclusion_agent",
    model=MODEL_NM,
    output_key="validated_exclusion_data",
    instruction="""
Given raw exclusion criteria output from exclusion_agent, format the output into a JSON structure with a key `patients` containing a list of patient dictionaries. Each patient should have the keys: id, matched_exclusion_criteria, and non_matched_exclusion_criteria.
"""
)

# Agent 1: Inclusion Agent using datastore
inclusion_agent = LlmAgent(
    name="inclusion_agent",
    model=MODEL_NM,
    tools=[vertex_search_tool],
    output_key="inclusion_data",
    instruction=f"""
You are a clinical trial assistant. Use the Vertex AI Search tool to retrieve patient documents from the datastore: {DATASTORE_ID}.
Identify patients satisfying inclusion criteria in the user query. Output patient ID, satisfied criteria, and missing criteria.
Then pass the results downstream.
"""
)

# Agent 2: Exclusion Agent using datastore
exclusion_agent = LlmAgent(
    name="exclusion_agent",
    model=MODEL_NM,
    tools=[vertex_search_tool],
    output_key="exclusion_data",
    instruction=f"""
You are a clinical trial assistant. Use the Vertex AI Search tool to retrieve patient documents from the datastore: {DATASTORE_ID}.
Identify patients matching the exclusion criteria. Output matched and non-matched criteria.
"""
)

# Optional: Summarization Agent
summarization_agent = LlmAgent(
    name="summarization_agent",
    model=MODEL_NM,
    tools=[vertex_search_tool],
    output_key="summary_data",
    instruction=f"""
You are a medical summarization assistant. Use the Vertex AI Search tool to read patient documents and provide a brief medical history summary for each patient ID.
"""
)

# Optional: Flagging Agent
flagging_agent = LlmAgent(
    name="flagging_agent",
    model=MODEL_NM,
    tools=[vertex_search_tool],
    output_key="flags_data",
    instruction=f"""
You are a quality control agent. Review patient documents using Vertex AI Search and flag any inconsistencies or conflicting medical history elements that could affect eligibility.
"""
)

# Matching + Validation Parallel
matching_parallel_agent = ParallelAgent(
    name="parallel_matching_agent",
    sub_agents=[inclusion_agent, exclusion_agent],
    description="Parallel agent to evaluate inclusion and exclusion."
)

# Validation Agent block
validation_parallel_agent = ParallelAgent(
    name="validation_agent_group",
    sub_agents=[validate_inclusion_agent, validate_exclusion_agent],
    description="Parallel agent to validate and format outputs of inclusion and exclusion agents."
)

# Reporting Agent
reporting_agent = LlmAgent(
    name="reporting_agent",
    model=MODEL_NM,
    tools=[generate_patient_reports],
    instruction="Generate the final patient eligibility report in JSON format using the validated inclusion and exclusion outputs."
)

# Final Sequential Workflow
workflow_agent = SequentialAgent(
    name="clinical_trial_sequential_workflow",
    sub_agents=[matching_parallel_agent, validation_parallel_agent, reporting_agent],
    description="Runs matching first, validates outputs, then generates final reports."
)

root_agent=workflow_agent
# Ready for CLI execution via: adk run multi_tool_agent
