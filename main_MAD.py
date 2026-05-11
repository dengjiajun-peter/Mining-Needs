import json
from record import base_path
import re
import random
from model import *
from Dataset import *
from record import *
# from PIL import Image
from map_raw_id import RawIdMapper
import os
import argparse


FINAL_ONLY_REASONINGS = {"io_pos", "io_posneg", "kg", "carp_json", "dtg_json"}
HOMOGENEOUS_DEBATE_REASONINGS = {"io_pos", "io_posneg", "kg", "dtg_json", "carp_json"}

# ------------------------------------------------------------
# Prompts
# ------------------------------------------------------------
eval_prompt = (
    "Question: {question}. There are the solutions to the question from different agents.\n"
    "Solution 1: {solution1}\n"
    "Solution 2: {solution2}\n"
    "Solution 3: {solution3}\n"
)

json_format_prompt = (
    'Please choose the best solution and output your answer in JSON format, with the format as follows: '
    '{"Reason": "", "Index": ""}. "Index" in the format should only be the index number of the right solution. '
    'Please strictly output in JSON format, do not output irrelevant content.'
)


def construct_message(agents, question, idx, reasoning):
    
    prefix_string = "These are other agents' process + answer messages using different reasoning methods: "
    constraint = "\n do not output irrelevant content."
    for agent in agents:
        agent_response = agent[idx]
        response = "\n\n One agent message (process + answer): ```{}```".format(agent_response)
        
        prefix_string = prefix_string + response 

    # Extract plain review text when the question contains a full prompt block.
    question_text_only = question
    if isinstance(question, str):
        m = re.search(r"Review Text:\s*\n?(.*?)(?:\n\n|\Z)", question, re.S)
        if m:
            question_text_only = m.group(1).strip()
        else:
            question_text_only = question.strip()

    if reasoning == 'io':
        prefix_string = prefix_string + """\n\n Using the answers of different methods as additional information, can you provide your answer to the question? \n {}""".format(question)
    elif reasoning == 'io_pos':
        prefix_string = prefix_string + """\n\n Use the other agents' methods as additional information, then answer with the SAME positive-shot criteria and output format as the original prompt below.
 {}""".format(question)
    elif reasoning == 'io_posneg':
        prefix_string = prefix_string + """\n\n Use the other agents' methods as additional information, then answer with the SAME positive-negative-shot criteria and output format as the original prompt below.
 {}""".format(question)
    elif reasoning == 'kg':
        prefix_string += """\n\n
Use the other agents'  methods as additional information.
Do NOT repeat the knowledge extraction stage.
Based on the review and the other agents' final judgments, directly revise your final decision.

Output strictly in this format:
Label: <0 or 1>
Novel Feature: <brief feature name, or None>
Because: <short reason>

Review Text:
{}
""".format(question_text_only)
    elif reasoning == 'carp_json':
        prefix_string = prefix_string + """\n\n Use the other agents' methods as additional information, then answer with the SAME CARP-JSON criteria and output schema as the original prompt below.
 {}""".format(question)
    elif reasoning == 'dtg_json':
        prefix_string = prefix_string + """\n\n Use the other agents' methods as additional information, then answer with the SAME DTG-JSON criteria and output schema as the original prompt below.
Do diagnosis internally and do not output intermediate diagnosis steps.
 {}""".format(question)
    elif reasoning == 'cot':
        prefix_string = prefix_string + """\n\n Using the answers of different methods as additional information, can you provide your answer to the question? \n {}
Please strictly follow the format below. First, let's think step by step. Write your reasoning inside <reasoning> tags by answering these 3 questions:
1. What specific product features does the reviewer mention?
2. Are these features currently available in existing products on the market?
3. Does the reviewer express a desire or hope for something that doesn't exist yet?

After your <reasoning> block, provide your final answer using the strictly requested format.""".format(question)
    elif reasoning == 'ccot':
        prefix_string += """\n\n Using the answers of different methods as additional information, generate a scene graph in JSON format for the provided image and its associated question.
{}

The scene graph should include:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Obect relationships that are relevant to answering the question.

Just generate the scene graph in JSON format. Do not say extra words.""".format(question)
    elif reasoning == 'sbp':
        prefix_string += """\n\n Using the answers of different methods as additional information, apply step-back prompting: first identify the broader unmet user need, then determine whether the review expresses a genuinely novel future product feature or unmet customer need.\n {}\n""".format(question)
    elif reasoning == 'ddcot':
        prefix_string += '''Using the answers of different methods as additional information, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions. Then with the aim of helping humans answer the original question, try to answer the sub-questions. 
{}
        
The expected answering form is as follows:
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...

Sub-answers:
1. <sub-answer 1>
2. <sub-answer 2>
...'''.format(question)

    return prefix_string + constraint



def construct_assistant_message(model, completion):
    # Ensure model string checking is done correctly
    if isinstance(model, str) and (('gpt' in model) or ('llama' in model) or ('gemini' in model)):
        try:
            content = completion["choices"][0]["message"]["content"]
            return {"role": "assistant", "content": content}
        except Exception:
            return None


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def extract_final_judgment(output_text):
    """Extract only Label / Novel Feature / Because for debate-time context."""
    if output_text is None:
        return ""

    text = str(output_text).strip()
    label = None
    novel_feature = None
    because = None

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if label is None and lower.startswith("label:"):
            label = stripped.split(":", 1)[1].strip()
        elif novel_feature is None and lower.startswith("novel feature:"):
            novel_feature = stripped.split(":", 1)[1].strip()
        elif because is None and lower.startswith("because:"):
            because = stripped.split(":", 1)[1].strip()

    if label is None:
        m = re.search(r"(?i)the answer is\s*([01U])", text)
        if m:
            label = m.group(1)

    if label is None:
        return text

    if novel_feature is None:
        novel_feature = "None"
    if because is None:
        because = "Not provided."

    return f"Label: {label}\nNovel Feature: {novel_feature}\nBecause: {because}"


ccot_prompt = '''
First, get the scene graph of the image in JSON format:
{scene_graph}
Then, use the image and scene graph as context to answer the question.
{answer}
'''


ddcot_prompt = '''
First, the problem can be deconstructed down to sub-questions.
{subquestion_answers}
Then, according to the sub-questions and sub-answers to answer the question.
{answer}
'''


custom_judge_prompt = (
    'Here are three candidate answers for the same review text, each produced by a different reasoning method.\n'
    'Review Text:\n'
    '{text}\n'
    'Candidate 1 (Direct reasoning):\n'
    '{solution1}\n'
    'Candidate 2 (Aspect-graph reasoning):\n'
    '{solution2}\n'
    'Candidate 3 (Sub-question reasoning):\n'
    '{solution3}\n'
    'Compare the three candidates. Choose the one with the most consistent and well-supported reasoning.\n'
    'Output ONLY a JSON object in this exact format, no extra text:\n'
    '{{"Reason": "<one sentence why>", "Index": "<1, 2, or 3>"}}\n'
)


if __name__ == "__main__":
    # 迁移 main_DMAD.py 的多 agent 断点续传、变量、保存等主流程，适配 MAD 只用 IO 辩论
    agents = 3
    rounds = 6
    random.seed(0)

  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--dataset", type=str, default="ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lecture", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--device", type=str, default = "cuda")
    parser.add_argument("--max_new_tokens", type=int, default = 8192)
    parser.add_argument("--google_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_base_url", type=str, default = "xxx")
    parser.add_argument("--ollama_base_url", type=str, default = "http://localhost:11434")
    parser.add_argument("--max_samples", type=int, default = -1, help="limit number of samples, -1 means all")
    parser.add_argument("--custom_data_path", type=str, default="", help="path to custom Excel/CSV file (used when --dataset custom)")
    parser.add_argument("--discussion_mode", action='store_true', help="If set, from round 2 onward agents only output Label: 0/1 (no process).")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix to append to DMAD output filenames (no leading underscore).")
    parser.add_argument("--vllm_api_key", type=str, default = "", help="vLLM OpenAI-compatible server API key. Empty means fallback to --openai_api_key / env.")
    parser.add_argument("--vllm_base_url", type=str, default = "", help="vLLM OpenAI-compatible server base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--reasoning", type=str, default="io", help="Reasoning method: io, io_pos, io_posneg, kg, carp_json, dtg_json, cot, sbp")

    args = parser.parse_args()    
    if getattr(args, "vllm_base_url", ""):
        args.openai_base_url = args.vllm_base_url
    if getattr(args, "vllm_api_key", ""):
        args.openai_api_key = args.vllm_api_key


    args.model_base = None
    args.model_name = None
    args.conv_mode = None
    args.sep = ","
    args.top_p = None
    args.system = None
    args.messages = None
    args.query = None

    
    client, args.tokenizer, args.llava, args.image_processor, args.gemini_model = MLLM_load(args)
    
    generated_description = []



    # 根据 reasoning 参数分支，分开逻辑
    safe_model = get_safe_model_label(args.model)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    if args.discussion_mode:
        suffix += "_discussion"
    if args.reasoning in ["io", "io_pos", "io_posneg", "kg", "carp_json", "dtg_json"]:
        base_reasoning = args.reasoning
        debate_reasoning = base_reasoning if base_reasoning in HOMOGENEOUS_DEBATE_REASONINGS else "io"
        final_only_context = base_reasoning in FINAL_ONLY_REASONINGS

        base_outputs = read_record(args.dataset, args.model, base_reasoning, split = args.split, lecture = args.lecture)
        if args.dataset == "custom":
            base_out = os.path.join(os.getcwd(), "outputs_none_lectures")
            actual_model_name = "hf_unsloth_Meta-Llama-3.1-8B-bnb-4bit"
            reasoning_path = os.path.join(base_out, f"{args.dataset}_{args.split}_{actual_model_name}_{base_reasoning}.json")
            base_outputs = read_record(args.dataset, args.model, base_reasoning, path=reasoning_path)
            print(f"DEBUG: Found {len(base_outputs)} {base_reasoning.upper()} records")

        if base_reasoning == "io":
            output_path = os.path.join(base_path, f'outputs_DMAD/{args.dataset}_{args.split}_{safe_model}_{agents}_{rounds}{suffix}.json')
        else:
            output_path = os.path.join(base_path, f'outputs_DMAD/{args.dataset}_{args.split}_{safe_model}_{base_reasoning}_{agents}_{rounds}{suffix}.json')
        print(f"[DEBUG] Output path: {output_path}")

        existing_outputs = read_json_from_path(output_path)
        start_idx = 0
        if isinstance(existing_outputs, list) and len(existing_outputs) > 0:
            generated_description = existing_outputs
            start_idx = len(existing_outputs)
        else:
            generated_description = []

        for n in range(start_idx, len(base_outputs)):
            example = base_outputs[n]
            if args.dataset == "ScienceQA":
                choices  = example['choices']
                hint = example['hint']
                answer   = example['choices'][example['answer']]
                question = create_prompt(question, choices, context = hint)
                args.image_path = os.path.join(base_path, f"dataset/ScienceQA/{n}.png")
            elif args.dataset == "mm-vet":
                question = create_prompt(question)
                imagename = base_outputs[n]['imagename']
                args.image_path = os.path.join(base_path, 'dataset/mm-vet/images', imagename)
            elif args.dataset == "custom":
                text = str(example["text"])
                if base_reasoning == "io":
                    question = create_custom_io_prompt(text)
                elif base_reasoning == "io_pos":
                    question = create_custom_io_pos_prompt(text)
                elif base_reasoning == "io_posneg":
                    question = create_custom_io_posneg_prompt(text)
                elif base_reasoning == "kg":
                    question = create_custom_kg_prompt(text)
                elif base_reasoning == "carp_json":
                    question = create_custom_carp_json_prompt(text)
                elif base_reasoning == "dtg_json":
                    question = create_custom_dtg_json_prompt(text)
                else:
                    question = create_custom_prompt(text)
                if 'raw_id' in example:
                    example['_raw_id'] = example['raw_id']
                if 'num' in example:
                    example['_num'] = example['num']
                args.image_path = None
                args.image_file = None

            example[f'{base_reasoning}_output'] = base_outputs[n]['outputs']
            args.image_path = None
            args.image_file = None
            agent_contexts = [[question] for agent in range(agents)]
            answers_ = [[] for round in range(rounds)]

            for r in range(rounds):
                for i, agent_context in enumerate(agent_contexts):
                    if r == 0:
                        outputs_list = example.get('outputs', []) if isinstance(example, dict) else []
                        base_output = outputs_list[i] if (i < len(outputs_list) and outputs_list[i] is not None) else ""
                        answer = base_output
                        debate_output = extract_final_judgment(base_output) if final_only_context else base_output
                        assistant_message = 'Directly answer the question. ' + debate_output
                        agent_context.append(assistant_message)
                        answers_[r].append(answer)
                    else:
                        others = agent_contexts[:i] + agent_contexts[i+1:]
                        message = construct_message(others, question, 2*r - 1, debate_reasoning)
                        args.messages = agent_context + [message]
                        generated = MLLM_generate(args)
                        answer = generated
                        debate_output = extract_final_judgment(generated) if final_only_context else generated
                        assistant_message = 'Directly answer the question. ' + debate_output
                        agent_context.append(message)
                        agent_context.append(assistant_message)
                        answers_[r].append(answer)

            example['agent_contexts'] = agent_contexts
            example['answers_'] = answers_
            example['usage'] = get_usage(args.model)
            if "outputs" in example:
                del example["outputs"]
            generated_description.append(example)
            with open(output_path, 'w') as f:
                json.dump(generated_description, f, indent=4)
        print(f"[MAD] Saved {base_reasoning.upper()} results to {output_path}")
    elif args.reasoning == "cot":
        cot_outputs = read_record(args.dataset, args.model, "cot", split=args.split, lecture=args.lecture)
        if args.dataset == "custom":
            base_out = os.path.join(os.getcwd(), "outputs_none_lectures")
            actual_model_name = "hf_unsloth_Meta-Llama-3.1-8B-bnb-4bit"
            cot_path = os.path.join(base_out, f"{args.dataset}_{args.split}_{actual_model_name}_cot.json")
            cot_outputs = read_record(args.dataset, args.model, "cot", path=cot_path)
            print(f"DEBUG: Found {len(cot_outputs)} COT records")
        cot_output_path = os.path.join(base_path, f'outputs_DMAD/{args.dataset}_{args.split}_{safe_model}_cot_{agents}_{rounds}{suffix}.json')
        print(f"[DEBUG] COT Output path: {cot_output_path}")
        existing_cot_outputs = read_json_from_path(cot_output_path)
        cot_generated_description = []
        cot_start_idx = 0
        if isinstance(existing_cot_outputs, list) and len(existing_cot_outputs) > 0:
            cot_generated_description = existing_cot_outputs
            cot_start_idx = len(existing_cot_outputs)
        else:
            cot_generated_description = []
        for n in range(cot_start_idx, len(cot_outputs)):
            example = cot_outputs[n]
            if args.dataset == "ScienceQA":
                choices = example['choices']
                hint = example['hint']
                answer = example['choices'][example['answer']]
                question = create_prompt(question, choices, context=hint)
                args.image_path = os.path.join(base_path, f"dataset/ScienceQA/{n}.png")
            elif args.dataset == "mm-vet":
                question = create_prompt(question)
                imagename = cot_outputs[n]['imagename']
                args.image_path = os.path.join(base_path, 'dataset/mm-vet/images', imagename)
            elif args.dataset == "custom":
                text = str(example["text"])
                question = create_custom_prompt(text)
                if 'raw_id' in example:
                    example['_raw_id'] = example['raw_id']
                if 'num' in example:
                    example['_num'] = example['num']
                args.image_path = None
                args.image_file = None
            example['cot_output'] = cot_outputs[n]['outputs']
            args.image_path = None
            args.image_file = None
            agent_contexts = [[question] for agent in range(agents)]
            answers_ = [[] for round in range(rounds)]
            for r in range(rounds):
                for i, agent_context in enumerate(agent_contexts):
                    if r == 0:
                        outputs_list = example.get('outputs', []) if isinstance(example, dict) else []
                        base_output = outputs_list[i] if (i < len(outputs_list) and outputs_list[i] is not None) else ""
                        assistant_message = base_output
                        answer = base_output
                        agent_context.append(assistant_message)
                        answers_[r].append(answer)
                    else:
                        others = agent_contexts[:i] + agent_contexts[i+1:]
                        message = construct_message(others, question, 2*r - 1, 'cot')
                        args.messages = agent_context + [message]
                        assistant_message = MLLM_generate(args)
                        answer = assistant_message
                        assistant_message = assistant_message
                        agent_context.append(message)
                        agent_context.append(assistant_message)
                        answers_[r].append(answer)
            example['agent_contexts'] = agent_contexts
            example['answers_'] = answers_
            example['usage'] = get_usage(args.model)
            if "outputs" in example:
                del example["outputs"]
            cot_generated_description.append(example)
            with open(cot_output_path, 'w') as f:
                json.dump(cot_generated_description, f, indent=4)
        print(f"[MAD] Saved COT results to {cot_output_path}")
    elif args.reasoning == "sbp":
        sbp_outputs = read_record(args.dataset, args.model, "sbp", split=args.split, lecture=args.lecture)
        if args.dataset == "custom":
            base_out = os.path.join(os.getcwd(), "outputs_none_lectures")
            actual_model_name = "hf_unsloth_Meta-Llama-3.1-8B-bnb-4bit"
            sbp_path = os.path.join(base_out, f"{args.dataset}_{args.split}_{actual_model_name}_sbp.json")
            sbp_outputs = read_record(args.dataset, args.model, "sbp", path=sbp_path)
            print(f"DEBUG: Found {len(sbp_outputs)} SBP records")
        sbp_output_path = os.path.join(base_path, f'outputs_DMAD/{args.dataset}_{args.split}_{safe_model}_sbp_{agents}_{rounds}{suffix}.json')
        print(f"[DEBUG] SBP Output path: {sbp_output_path}")
        existing_sbp_outputs = read_json_from_path(sbp_output_path)
        sbp_generated_description = []
        sbp_start_idx = 0
        if isinstance(existing_sbp_outputs, list) and len(existing_sbp_outputs) > 0:
            sbp_generated_description = existing_sbp_outputs
            sbp_start_idx = len(existing_sbp_outputs)
        else:
            sbp_generated_description = []
        for n in range(sbp_start_idx, len(sbp_outputs)):
            example = sbp_outputs[n]
            if args.dataset == "ScienceQA":
                choices = example['choices']
                hint = example['hint']
                answer = example['choices'][example['answer']]
                question = create_prompt(question, choices, context=hint)
                args.image_path = os.path.join(base_path, f"dataset/ScienceQA/{n}.png")
            elif args.dataset == "mm-vet":
                question = create_prompt(question)
                imagename = sbp_outputs[n]['imagename']
                args.image_path = os.path.join(base_path, 'dataset/mm-vet/images', imagename)
            elif args.dataset == "custom":
                text = str(example["text"])
                question = create_custom_prompt(text)
                if 'raw_id' in example:
                    example['_raw_id'] = example['raw_id']
                if 'num' in example:
                    example['_num'] = example['num']
                args.image_path = None
                args.image_file = None
            example['sbp_output'] = sbp_outputs[n]['outputs']
            args.image_path = None
            args.image_file = None
            agent_contexts = [[question] for agent in range(agents)]
            answers_ = [[] for round in range(rounds)]
            for r in range(rounds):
                for i, agent_context in enumerate(agent_contexts):
                    if r == 0:
                        outputs_list = example.get('outputs', []) if isinstance(example, dict) else []
                        base_output = outputs_list[i] if (i < len(outputs_list) and outputs_list[i] is not None) else ""
                        assistant_message = base_output
                        answer = base_output
                        agent_context.append(assistant_message)
                        answers_[r].append(answer)
                    else:
                        others = agent_contexts[:i] + agent_contexts[i+1:]
                        message = construct_message(others, question, 2*r - 1, 'sbp')
                        args.messages = agent_context + [message]
                        assistant_message = MLLM_generate(args)
                        answer = assistant_message
                        agent_context.append(message)
                        agent_context.append(assistant_message)
                        answers_[r].append(answer)
            example['agent_contexts'] = agent_contexts
            example['answers_'] = answers_
            example['usage'] = get_usage(args.model)
            if "outputs" in example:
                del example["outputs"]
            sbp_generated_description.append(example)
            with open(sbp_output_path, 'w') as f:
                json.dump(sbp_generated_description, f, indent=4)
        print(f"[MAD] Saved SBP results to {sbp_output_path}")
