"""System prompts for the clinical inference pipeline.

Every system prompt / agent instruction string used by this project
lives here as a named constant, so the agent's instructions are
visible and inspectable in one place — never buried inline in agent
logic. Agent code imports these constants; it does not define its own.
"""

from __future__ import annotations

# The Strands conversational agent's system prompt (agent.build_agent):
# the agent wraps the deterministic pipeline and writes message text
# only — it never decides a pipeline transition.
SYSTEM_PROMPT = (
    "You are the assistant for the clinical inference pipeline. "
    "You can run the end-to-end pipeline on a file, list the registered "
    "models, and summarize a completed run. You never see raw patient "
    "data: the pipeline anonymizes everything before any model or LLM "
    "call. Answer concisely, in plain language, and always name the "
    "run directory and the model that was used."
)

# The live model-pick prompt (llm.LLMClient.choose_model), in two
# parts: the instruction prefix and the reply-format suffix. The
# candidate list is interpolated between them at call time.
MODEL_PICK_PROMPT_PREFIX = (
    "You are choosing the best clinical ML model for a dataset. "
    "Eligible models (all required features matched):\n"
)
MODEL_PICK_PROMPT_SUFFIX = (
    "\nReply with exactly two lines:\n"
    "MODEL: <model name>\nJUSTIFICATION: <one sentence>"
)

# The live image-header prompt (llm.LLMClient.describe_image): the
# multi-stage modality/disease classification (the paper's MedGemma
# step), sent with the image itself.
IMAGE_PROMPT = (
    "What imaging modality is this, and what disease type (if any) "
    "is visible? Reply with exactly two lines:\n"
    "MODALITY: <modality>\nDISEASE: <disease type or none>"
)

__all__ = [
    "IMAGE_PROMPT",
    "MODEL_PICK_PROMPT_PREFIX",
    "MODEL_PICK_PROMPT_SUFFIX",
    "SYSTEM_PROMPT",
]
