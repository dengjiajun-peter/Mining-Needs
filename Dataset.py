from pydoc import text

from datasets import load_dataset
import os
import json
import pandas as pd

from record import base_path


def read_dataset(name, split):
    if name == "ScienceQA":
        data = load_dataset("derek-thomas/ScienceQA", split = split)
    elif name == "mm-vet":
        meta_data = os.path.join(base_path, "dataset/mm-vet/mm-vet.json")
        with open(meta_data, "r") as f:
            data = json.load(f)
    elif name == "custom":
        # split is used as the file path to the Excel/CSV file
        file_path = split
         
            # 支持 custom_data_path 为 .json 文件
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                return json.load(f)
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        # Preserve original row identity from source file.
        if "raw_id" not in df.columns:
            if "id" in df.columns:
                df["raw_id"] = df["id"]
            elif "ID" in df.columns:
                df["raw_id"] = df["ID"]
            elif "num" in df.columns:
                df["raw_id"] = df["num"]
            else:
                df["raw_id"] = df.index
        data = df.to_dict(orient="records")
    return data 


def create_options(options):
    if options != None:
        letters = ["(A) ", "(B) ", "(C) ", "(D) ", "(E) ", "(F) ", "(G) "]
        strs = "Options:\n"
        for i in range(len(options)):
            strs += letters[i]
            strs += options[i]
            strs += "\n"
        strs += "\n"
    else:
        strs = ""
    return strs


def create_lecture(lecture = None):
    strs = ""
    if lecture != None:
        strs = "Lecture:\n" + lecture + "\n\n"
    return strs


def create_context(context = None):
    strs = ""
    if context != None and context != "":
        strs = "Context:\n" + context + "\n\n"
    return strs


## ──────────────────────────────────────────────
## DTG prompts for custom dataset
## DTG = Deliberate then Generate / Deliberate then Decide
## We adapt it to binary novelty classification:
## first detect likely error type, then output refined label.
## ──────────────────────────────────────────────

def create_custom_dtg_base_prompt(text):
    """
    DTG-style base prompt adapted for binary novelty classification.
    Following DTG:
    input -> empty candidate -> detect error type -> refined final decision
    """
    prompt = "Review Text:\n" + text + "\n\n"

    prompt += (
        "Task: Determine whether the review describes a novel future feature / unmet customer need.\n\n"
        "Novel customer needs are defined as requirements that:\n"
        "1) are currently NOT supported by existing product features or have not been previously included in the product line; AND\n"
        "2) are implicitly or explicitly expressed by the reviewer.\n"
        "If either condition is missing, Label = 0.\n\n"
    )

    prompt += (
        "The preliminary decision is:\n"
        "\n"
        "\n"
        "Please detect the error type firstly, and provide the refined final decision then.\n\n"
    )

    prompt += (
        "Possible error types may include:\n"
        "- existing_feature_described_as_future_need\n"
        "- standard_complaint_mistaken_as_novel_feature\n"
        "- purchase_preference_mistaken_as_novel_feature\n"
        "- vague_positive_description_mistaken_as_novel_feature\n"
        "- missed_future_desire_signal\n"
        "- missed_currently_unsupported_signal\n"
        "- no_error\n\n"
    )

    prompt += (
        "Output strictly in the following format:\n\n"
        "[Error Type]\n"
        "<one short phrase>\n\n"
        "[Refined Label]\n"
        "<0 or 1>\n\n"
        "[Reasoning]\n"
        "<brief reasoning in no more than 2 sentences>\n"
    )

    return prompt


def create_custom_dtg_negdemo_prompt(text):
    """
    DTG-style prompt with one negative demonstration and strict JSON output.
    The model should still diagnose error type, but return only a JSON object.
    """
    prompt = (
        "You are an expert system for diagnosing customer reviews.\n\n"
        "Task:\n"
        "Determine whether a review expresses a novel future feature or unmet customer need.\n\n"
        "Novel customer needs are requirements that:\n"
        "1) are currently NOT supported by existing product features or have not been previously included in the product line; AND\n"
        "2) are implicitly or explicitly expressed by the reviewer.\n"
        "If either condition is missing, label = 0.\n\n"
    )

    # Negative demonstration
    prompt += (
        "Example:\n"
        "Review Text:\n"
        "Disappointed with the screen quality. The picture is the quality of an 80s TV and there is such a huge glare that it is hard to use during the day. Would never purchase again!\n\n"
        "Correct JSON output:\n"
        "{\n"
        "  \"error_type\": \"standard_complaint_mistaken_as_novel_feature\",\n"
        "  \"label\": 0,\n"
        "  \"novel_feature\": null,\n"
        "  \"reason\": \"The review only complains about current screen quality and glare, which are existing performance issues. It does not request a new future feature or a currently unsupported capability.\"\n"
        "}\n\n"
    )

    # Test instance
    prompt += (
        "Now solve the following case.\n\n"
        "Review Text:\n" + text + "\n\n"

        "Possible error_type values may include:\n"
        "- existing_feature_described_as_future_need\n"
        "- standard_complaint_mistaken_as_novel_feature\n"
        "- purchase_preference_mistaken_as_novel_feature\n"
        "- vague_positive_description_mistaken_as_novel_feature\n"
        "- missed_future_desire_signal\n"
        "- missed_currently_unsupported_signal\n"
        "- no_error\n\n"

        "Important decision rules:\n"
        "1) If the review describes an existing feature, existing specification, existing design, or current capability, label = 0.\n"
        "2) If the review is only a complaint about current quality, performance, warranty, delivery, refund, compatibility, or usability, label = 0.\n"
        "3) If the review only expresses purchase preference, satisfaction, or a requirement already satisfied by the current product, label = 0.\n"
        "4) Output label = 1 only when the review clearly expresses a desired future feature or capability that is currently unavailable or unsupported.\n\n"

        "Output requirements:\n"
        "- Output ONLY one valid JSON object.\n"
        "- No markdown, no code fence, no extra text outside JSON.\n"
        "- label must be exactly 0 or 1.\n"
        "- If label = 0, novel_feature must be null.\n"
        "- If label = 1, novel_feature must be a short feature phrase from the review intent.\n\n"

        "JSON Schema:\n"
        "{\n"
        "  \"error_type\": string,\n"
        "  \"label\": 0 or 1,\n"
        "  \"novel_feature\": string or null,\n"
        "  \"reason\": string\n"
        "}\n"
    )

    return prompt


def create_custom_dtg_json_prompt(text):
    """
    DTG-style prompt with strict JSON output.
    Diagnosis is internal; only final decision is returned.
    """
    prompt = (
        "You are an expert system for diagnosing customer reviews.\n\n"
        "Task:\n"
        "Determine whether the review expresses a novel future feature or unmet customer need.\n\n"
        "Process:\n"
        "- Internally diagnose whether the review reflects a genuinely new unmet need.\n"
        "- Then generate the final decision.\n\n"
        "IMPORTANT:\n"
        "- Your diagnosis should NOT be shown.\n"
        "- Output ONLY a valid JSON object.\n\n"
        "JSON Schema:\n"
        "{\n"
        '  "label": 0 or 1,\n'
        '  "novel_feature": string or null,\n'
        '  "reason": string\n'
        "}\n\n"
        "Rules:\n"
        "- label must be exactly 0 or 1.\n"
        "- If label = 0, novel_feature must be null.\n"
        "- No extra text outside JSON.\n\n"
        "Review:\n"
        f"{text}\n"
    )
    return prompt



# ──────────────────────────────────────────────
# Custom dataset prompts (text-only, no image)
# Columns expected: label, novel_reason, text
# ──────────────────────────────────────────────
    
CUSTOM_ANSWER_FORMAT = '''Please answer in the following format:
Label: <0 or 1>  (1 = the review mentions a novel/not-yet-existing feature; 0 = it does not)
Novel Feature: <brief feature name, or "None" if label is 0>
Because: <your explanation>
'''

CARP_OUTPUT_TEXT_FORMAT = """
Output strictly in the following format:

[Label]
<0 or 1>

[Novel Feature]
<exact feature phrase if Label=1, otherwise None>

[Positive Evidence]
- <quoted phrase from the review>
- <quoted phrase from the review>
(or write exactly: None)

[Negative Evidence]
- <quoted phrase from the review>
- <quoted phrase from the review>
(or write exactly: None)

[Reasoning]
<brief reasoning in no more than 2 sentences>
"""

def create_custom_carp_prompt(text):
    """
    CARP-style evidence-first prompt for the custom dataset.
    Text version with strongly constrained section-based output.
    """
    prompt = "Review Text:\n" + text + "\n\n"

    prompt += (
        "Task: Determine whether the review describes a novel future feature / unmet customer need.\n\n"
        "Novel customer needs are defined as requirements that:\n"
        "1) are currently NOT supported by existing product features or have not been previously included in the product line; AND\n"
        "2) are implicitly or explicitly expressed by the reviewer.\n"
        "If either condition is missing, Label = 0.\n\n"
    )

    prompt += (
        "Please solve the task in the following steps.\n\n"

        "Step 1: Extract POSITIVE clues.\n"
        "Quote exact phrases from the review that suggest the reviewer wants a new or not-yet-existing feature.\n"
        "If no such clue exists, write None.\n\n"

        "Step 2: Extract NEGATIVE clues.\n"
        "Quote exact phrases from the review that indicate the review is only describing existing features, standard complaints, performance issues, or ordinary missing features.\n"
        "If no such clue exists, write None.\n\n"

        "Step 3: Diagnostic reasoning.\n"
        "Based ONLY on the extracted clues, determine:\n"
        "- what feature is being discussed,\n"
        "- whether it is currently unavailable / unsupported,\n"
        "- whether the reviewer clearly desires it for the future.\n\n"

        "Step 4: Final decision.\n"
        "If the review clearly expresses a desired future feature that is not currently supported, Label = 1.\n"
        "Otherwise, Label = 0.\n\n"
    )

    prompt += CARP_OUTPUT_TEXT_FORMAT
    return prompt

def create_custom_carp_json_prompt(text):
    """
    CARP-style evidence-first prompt with strict JSON output.
    The model may reason internally, but MUST output valid JSON only.
    """
    prompt = (
        "You are an expert analyst for identifying novel customer needs.\n\n"
        "Task:\n"
        "Determine whether the following review expresses a novel, future-oriented, unmet customer need.\n\n"
        "Definition:\n"
        "A novel customer need is a requirement that:\n"
        "1) Is NOT supported by existing product features; AND\n"
        "2) Is explicitly or implicitly expressed by the reviewer.\n\n"
        "Instructions:\n"
        "- First, internally extract positive and negative evidence from the review.\n"
        "- Then make a final decision.\n"
        "- DO NOT output your analysis.\n"
        "- Output ONLY a valid JSON object following the schema below.\n\n"
        "JSON Schema:\n"
        "{\n"
        '  "label": 0 or 1,\n'
        '  "novel_feature": string or null,\n'
        '  "reason": string\n'
        "}\n\n"
        "Rules:\n"
        "- If label = 0, novel_feature MUST be null.\n"
        "- If label = 1, novel_feature MUST be a short phrase describing the novel need.\n"
        "- The output must be valid JSON. No extra text.\n\n"
        "Review:\n"
        f"{text}\n"
    )
    return prompt

def format_other_agent_outputs(other_outputs):
    """
    Convert a list of other agents' outputs into a readable block.
    """
    block = ""
    for idx, out in enumerate(other_outputs, start=1):
        block += f"[Other Agent {idx} Output]\n{out}\n\n"
    return block
def create_custom_revision_prompt_critique(text, other_outputs):
    others_block = format_other_agent_outputs(other_outputs)

    prompt = ""
    prompt += "Review Text:\n"
    prompt += text + "\n\n"

    prompt += "Other agents have produced the following answers:\n\n"
    prompt += others_block + "\n"

    prompt += (
        "Your task is to revise your previous decision if necessary.\n"
        "However, you must NOT simply follow the majority or the strongest-sounding argument.\n\n"

        "TASK DEFINITION (STRICT):\n"
        "A novel future feature must satisfy BOTH conditions:\n"
        "1) The feature is NOT currently supported by existing products or standard configurations; AND\n"
        "2) The reviewer explicitly or implicitly expresses a desire or need for this feature in the future.\n"
        "If either condition is missing, Label MUST be 0.\n\n"

        "IMPORTANT CLARIFICATION:\n"
        "The following do NOT count as novel future features:\n"
        "- Descriptions or praise of existing features or specifications\n"
        "- Complaints about performance or quality issues (battery life, heat, noise, speed, screen, etc.)\n"
        "- Requests for better versions of standard features (more RAM, faster CPU, longer battery)\n"
        "- Missing but standard components (ports, adapters, accessories)\n\n"

        "REVISION GUIDELINES:\n"
        "Step 1: Examine the evidence cited in other agents' answers.\n"
        "Determine whether the evidence only describes existing features, specifications, or common issues.\n\n"

        "Step 2: Check whether the review clearly expresses a FUTURE-ORIENTED desire\n"
        "for a feature that does NOT currently exist. The feature must be concrete and specific.\n\n"

        "Step 3: Make your final decision:\n"
        "- If the judgment relies mainly on existing features or standard complaints, set Label = 0.\n"
        "- Only set Label = 1 if there is clear evidence of a non-existent future feature.\n\n"

        "Provide your final revised answer in the SAME format as before.\n\n"
    )

    prompt += CUSTOM_ANSWER_FORMAT
    return prompt

def create_custom_revision_prompt_anticonform(text, other_outputs):
    """
    Anti-conformity revision prompt:
    explicitly prevents the agent from changing its answer due to majority pressure.
    Revision is allowed only when explicit contradictory textual evidence is found.
    """
    others_block = format_other_agent_outputs(other_outputs)

    prompt = "Review Text:\n" + text + "\n\n"
    prompt += "Other agents have produced the following answers:\n\n"
    prompt += others_block

    prompt += (
        "You must act as an independent reviewer.\n"
        "Do NOT change your answer merely because other agents agree with each other.\n\n"

        "Please follow these rules strictly:\n"
        "1. Treat consensus as irrelevant unless it is backed by explicit textual evidence.\n"
        "2. Only revise your answer if another agent provides a quoted phrase from the review that directly contradicts your current judgment.\n"
        "3. If the other agents only provide general reasoning without stronger evidence, keep your original judgment.\n"
        "4. Your decision must be grounded in local textual evidence, not social agreement.\n\n"

        "Before revising, silently check:\n"
        "- Is there a concrete phrase in the review that I missed?\n"
        "- Does that phrase clearly indicate a novel future feature?\n"
        "- Is the feature truly not supported by current products?\n"
        "- Is the reviewer clearly requesting or desiring it?\n\n"

        "Then provide your final revised answer in the format below.\n\n"
    )

    prompt += CUSTOM_ANSWER_FORMAT
    return prompt
def build_oneshot_block() -> str:
    return """
“To ground the LLM’s interpretation of the task, we include one positive and one negative reference example drawn from the dataset. These examples serve solely as illustrative anchors for applying the theoretical definition of novel customer needs, rather than as decision rules or heuristics. The reference samples are excluded from downstream evaluation.”
One-shot reference examples (follow the TASK DEFINITION strictly):

[Positive Example]
Review Text:
The laptop works well for basic tasks, but as a digital artist, I'm constantly frustrated by the physical constraints. I really wish someone would invent a laptop with a detachable, rollable e-ink secondary display that I could pull out from the side for drawing, instead of having to carry a bulky separate drawing tablet everywhere.
Correct Output:
Label: 1
Novel Feature: Detachable, rollable e-ink secondary display for drawing
Because: The review clearly expresses a desire for a future, non-existent hardware form factor (rollable side e-ink display) that solves a specific unmet need. It is not complaining about existing specs, but rather inventing a new physical feature.

[Negative Example]
Review Text:
I bought this laptop for school and it's definitely an upgrade from my 7-year-old PC. The i9 processor and RTX graphics handle my software easily. However, I am extremely disappointed with a few things. First, the battery life is atrocious—it dies in 3 hours even on light use, unlike the 8 hours advertised. Second, the fan noise is incredibly loud when gaming, and the upper side of the keyboard gets painfully hot. Also, for a $2000 machine, only having 16GB of RAM and no native DisplayPort is a joke. I had to buy a separate hub just to connect my monitors. The touchpad is also jumpy. Overall, decent performance, but the heating and battery issues make it hard to recommend.
Correct Output:
Label: 0
Novel Feature: None
Because: This review extensively details existing product configurations, performance metrics, and common flaws (battery life, fan noise, heat, RAM size, port availability, and trackpad issues). Complaining about missing standard ports (DisplayPort), insufficient RAM, or poor battery performance does NOT constitute a request for a "novel/future" feature. These are standard quality and specification complaints.
"""
def create_custom_prompt(text):
    prompt = "Review Text:\n" + text + "\n\n"
    prompt += "Task: Determine whether the review describes a product feature that does not yet exist in current products (a novel/future feature).\n\n"
  
    prompt += (
        "Task: Determine whether the review describes a novel future feature / unmet customer need.\n\n"
        "Novel customer needs are defined as new and unmet requirements that:\n"
        "1) are currently NOT supported by existing product features or have not been previously included in the product line; AND\n"
        "2) are implicitly or explicitly expressed by the reviewer (typically by lead users) within unstructured reviews.\n"
        "If either condition is missing → Label = 0.\n\n"
    )

    prompt += CUSTOM_ANSWER_FORMAT
    return prompt

def create_custom_io_prompt(text):
    """IO (direct answer) prompt for the custom dataset."""
    prompt = "Review Text:\n" + text + "\n\n"
    prompt += "Task: Determine whether the review describes a product feature that does not yet exist in current products (a novel/future feature).\n\n"
    prompt += CUSTOM_ANSWER_FORMAT
    return prompt


def create_custom_io_pos_prompt(text):
    """IO prompt with one positive shot for custom dataset."""
    prompt = """
[Positive Example]
Review Text:
Love this. Works great for me. A UI-driven handheld gadget for controlling smart home appliances

Correct Output:
Label: 1
Novel Feature: UI-driven handheld gadget for controlling smart home appliances
Because: The review explicitly describes a new type of device for controlling smart home appliances, which does not exist in current product offerings.

"""
    prompt += "Review Text:\n" + text + "\n\n"
    prompt += "Task: Determine whether the review describes a product feature that does not yet exist in current products (a novel/future feature).\n\n"
    prompt += CUSTOM_ANSWER_FORMAT
    return prompt


def create_custom_io_posneg_prompt(text):
    """IO prompt with one positive and one negative shot for custom dataset."""
    prompt = """
[Positive Example]
Review Text:
Love this. Works great for me. A UI-driven handheld gadget for controlling smart home appliances

Correct Output:
Label: 1
Novel Feature: UI-driven handheld gadget for controlling smart home appliances
Because: The review explicitly describes a new type of device that does not currently exist.

[Negative Example]
Review Text:
Disappointed with the screen quality. The picture is the quality of an 80’s tv and there is such a huge glare that it’s hard to use during the day. Would never purchase again!

Correct Output:
Label: 0
Novel Feature: None
Because: This review only complains about display quality and usability issues, which are problems with existing features rather than requests for novel future features.

"""
    prompt += "Review Text:\n" + text + "\n\n"
    prompt += "Task: Determine whether the review describes a product feature that does not yet exist in current products (a novel/future feature).\n\n"
    prompt += CUSTOM_ANSWER_FORMAT
    return prompt


def create_custom_kg_prompt(text):
    """Knowledge-generation style prompt without demonstrations."""
    prompt = "Review Text:\n" + text + "\n\n"
    prompt += """
Step 1: Knowledge Extraction
Extract any statements that indicate:
- a desired future feature
- a feature that does not currently exist

List them as bullet points.
If none exist, output "None".

Step 2: Decision
Based ONLY on the extracted knowledge above,
determine whether the review expresses a novel future feature.

"""
    prompt += CUSTOM_ANSWER_FORMAT
    return prompt


def create_custom_carp_revision_prompt(text, other_outputs):
    """
    CARP-style revision prompt for multi-agent debate.
    Agent sees other agents' 4-step evidence extraction and revises based on comparative evidence.
    """
    def format_other_agent_carp(out):
        """Extract the key parts from CARP output for comparison."""
        return out
    
    others_block = ""
    for idx, out in enumerate(other_outputs):
        others_block += f"[Other Agent {idx} CARP Output]\n{out}\n\n"

    prompt = "Review Text:\n" + text + "\n\n"
    prompt += "Other agents have produced the following CARP-style evidence extractions:\n\n"
    prompt += others_block

    prompt += (
        "Your task is to review their evidence extraction and revise your own answer based on comparative analysis.\n\n"
        
        "Please follow these steps:\n\n"
        "Step 1: Examine the POSITIVE clues extracted by other agents. Are they also supported by the review text?\n"
        "Step 2: Examine the NEGATIVE clues extracted by other agents. Do they provide stronger disconfirming evidence?\n"
        "Step 3: Compare the DIAGNOSTIC REASONING of other agents with your own understanding.\n"
        "Step 4: Revise your label ONLY if other agents provide stronger or more comprehensive evidence than your current extraction.\n\n"
        
        "Important rules:\n"
        "- Do not change your answer just because multiple agents agree.\n"
        "- Only revise if another agent cites clues you missed or provides more convincing diagnostic logic.\n"
        "- Prioritize direct quoted phrases from the review over general impressions.\n\n"
        
        "Now provide your REVISED answer following the CARP structure:\n\n"
    )

    prompt += (
        "Step 1: Extract POSITIVE clues.\n"
        "Quote exact phrases from the review that suggest the reviewer wants a new or not-yet-existing feature.\n"
        "If no such clue exists, write None.\n\n"

        "Step 2: Extract NEGATIVE clues.\n"
        "Quote exact phrases from the review that indicate the review is only describing existing features, standard complaints, performance issues, or ordinary missing features.\n"
        "If no such clue exists, write None.\n\n"

        "Step 3: Diagnostic reasoning.\n"
        "Based ONLY on the extracted clues, determine:\n"
        "- what feature is being discussed,\n"
        "- whether it is currently unavailable / unsupported,\n"
        "- whether the reviewer clearly desires it for the future.\n\n"

        "Step 4: Final decision.\n"
        "If the review clearly expresses a desired future feature that is not currently supported, Label = 1.\n"
        "Otherwise, Label = 0.\n\n"
    )

    prompt += CARP_OUTPUT_TEXT_FORMAT
    return prompt


# 阶段一：生成 Step-back 问题（Abstraction）
def create_sbp_step1_prompt(text):
    return f"""You are an expert at identifying novel future features and unmet customer needs. 
(Note: Novel customer needs are defined as requirements that are currently NOT supported by existing product features AND are expressed by the reviewer.)

Your task is to take a step back and extract the HIGH-LEVEL CONCEPTS and principles from the given review. 
DO NOT make a final judgment yet; only extract the abstract key information.

Here is an example:
Review: "I love this smartwatch, but I really wish it could measure my blood sugar levels non-invasively during my workouts."
High-level Abstraction: 
1. Core customer need: Non-invasive continuous glucose monitoring (CGM).
2. Existing product support: Not supported by current mainstream smartwatch product lines.

Now process this review:
Review: {text}

High-level Abstraction:"""

# 阶段二：结合背景进行推理（Reasoning）
def create_sbp_step2_prompt(text, abstraction_result):
    return f"""You are an expert at identifying novel future features and unmet customer needs. 
(Note: Novel customer needs are defined as requirements that are currently NOT supported by existing product features AND are expressed by the reviewer.)

You are given a customer review and a set of high-level information (principles) involved in judging the novelty. 
Solve the problem step by step by following the principles.

Review: {text}
High-level Abstraction: 
{abstraction_result}

{CUSTOM_ANSWER_FORMAT}"""


# def create_custom_cot_prompt(text):
#     """CoT (Chain-of-Thought) prompt for the custom dataset."""
#     prompt = "Review Text:\n" + text + "\n\n"

#     prompt += (
#         "Task: Determine whether the review describes a novel future feature / unmet customer need.\n\n"
#         "Novel customer needs are defined as new and unmet requirements that:\n"
#         "1) are currently NOT supported by existing product features or have not been previously included in the product line; AND\n"
#         "2) are implicitly or explicitly expressed by the reviewer (typically by lead users) within unstructured reviews.\n"
#         "If either condition is missing → Label = 0.\n\n"
#     )
#     prompt +=build_oneshot_block()  # 添加 one-shot 参考示例
#     prompt += "Let's think step by step:\n"
#     # 将推理过程和答案分开要求
#     prompt +=  "Briefly justify your answer by addressing the following points:\n"
#     prompt += "1. What specific product features does the reviewer mention?\n"
#     prompt += "2. Are these features currently available in existing products on the market?\n"
#     prompt += "3. Does the reviewer express a desire or hope for something that doesn't exist yet?\n\n"
#     prompt += "Based on your step-by-step reasoning above, provide your final answer.\n\n"
#     prompt += CUSTOM_ANSWER_FORMAT
#     return prompt
def create_custom_cot_prompt(text):
    """CoT (Chain-of-Thought) prompt for the custom dataset."""
    prompt = "Review Text:\n" + text + "\n\n"

    prompt += (
        "Task: Determine whether the review describes a novel future feature / unmet customer need.\n\n"
        # "Novel customer needs are defined as new and unmet requirements that:\n"
        # "1) are currently NOT supported by existing product features or have not been previously included in the product line; AND\n"
        # "2) are implicitly or explicitly expressed by the reviewer (typically by lead users) within unstructured reviews.\n"
        # "If either condition is missing → Label = 0.\n\n"
    )

    prompt += "Let's think step by step:\n"
    prompt += CUSTOM_ANSWER_FORMAT
    return prompt




def create_custom_ccot_scene_graph_prompt(text):
    """CCoT step-1: generate an aspect-analysis JSON for the review text."""
    prompt = "Review Text:\n" + text + "\n\n"
    prompt += """Analyze the review text and generate an aspect graph in JSON format that includes:
1. Product aspects / features mentioned in the review.
2. Attributes of each aspect (e.g. existing, desired, novel, missing).
3. Relationships between aspects, if any.

Only output the JSON. Do not add extra words."""
    return prompt


def create_custom_ccot_answer_prompt(scene_graph, answer):
    """CCoT step-2: answer using the aspect graph as context."""
    prompt = "Aspect graph of the review (JSON):\n" + scene_graph + "\n\n"
    prompt += "Use the review text and the aspect graph as context.\n\n"
    prompt += "Review Text:\n" + answer + "\n\n"
    prompt += "Task: Determine whether the review describes a product feature that does not yet exist in current products (a novel/future feature).\n\n"
    prompt += CUSTOM_ANSWER_FORMAT
    return prompt






def create_custom_ddcot_subquestions_prompt(text):
    """DdCoT step-1: 生成固定维度的子问题（对齐任务本质）"""
    prompt = "Review Text:\n" + text + "\n\n"
    prompt += """Task: Determine whether the review describes a novel product feature (defined as: a feature that does NOT exist in current products, and the reviewer hopes/wants/needs it to be added in the future).
Please strictly follow the format below to generate 4 fixed sub-questions and their answers:

Sub-questions:
1. Does the reviewer mention a specific feature they hope/want/need to add to the product (look for keywords: would love, if only, hope, need, should have, in the future)?
2. Is this feature clearly NOT present in the current product described in the review?
3. Does the reviewer refer to this feature as something for the future (not existing now)?
4. Is this feature a specific, concrete feature (not a vague complaint like "better performance")?

Sub-answers:
1. [Answer with Yes/No, and quote 1 key phrase from the review to support]
2. [Answer with Yes/No, and quote 1 key phrase from the review to support]
3. [Answer with Yes/No, and quote 1 key phrase from the review to support]
4. [Answer with Yes/No, and quote 1 key phrase from the review to support]"""
    return prompt


def create_custom_ddcot_answer_prompt(subquestion_answers, answer):
    """DdCoT step-2: 基于子问题答案，按规则推导最终结果"""
    # 定义固定的答案格式（替代原_CUSTOM_ANSWER_FORMAT）
    ANSWER_FORMAT = """Output strictly in this format:
Label: [0 or 1, 1=novel/future feature, 0=no novel feature]
Novel Feature: [If Label=1, extract the exact feature phrase from the review; if Label=0, write "None"]
Because: [Explain with sub-answers and quote key phrases from the review, max 2 sentences]"""
    
    prompt = "Review Text:\n" + answer + "\n\n"
    prompt += "Sub-questions and Sub-answers:\n" + subquestion_answers + "\n\n"
    prompt += """Inference Rules (MUST FOLLOW):
- If Sub-answer 1=Yes AND Sub-answer 2=Yes → Label=1
- All other cases → Label=0

Task: Determine whether the review describes a product feature that does not yet exist in current products (a novel/future feature).
"""
    prompt += ANSWER_FORMAT
    return prompt




def create_prompt(question, options = None, context = None, lecture = None, if_options = True, post = True):
    prompt = "Question:\n" + question + "\n\n"
    if if_options and options != None:
        prompt += create_context(context)
        prompt += create_options(options)
        prompt += create_lecture(lecture)
        postfix = '''Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer is A, you should answer: 
The answer is A. 
Because ...
'''        
        if post:
            prompt += postfix
    return prompt
    
    