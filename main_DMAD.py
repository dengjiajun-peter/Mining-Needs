import json
import os
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

eval_prompt = '''Question: {question}. There are the solutions to the question from different agents. 
Solution 1: {solution1}
Solution 2: {solution2}
Solution 3: {solution3}
'''
json_format_prompt = '''Please choose the best solution and output your answer in JSON format, with the format as follows: {"Reason": "", "Index": ""}. "Index" in the format should only be the index number of the right solution. Please strictly output in JSON format, do not output irrelevant content.'''


def construct_message(agents, question, idx, reasoning):
    
    prefix_string = "These are other agents' process + answer messages using different reasoning methods: "
    constraint = "\n do not output irrelevant content."
    for agent in agents:
        agent_response = agent[idx]
        response = "\n\n One agent message (process + answer): ```{}```".format(agent_response)
        
        prefix_string = prefix_string + response 

    if reasoning == 'io':
        prefix_string = prefix_string + """\n\n Using the answers of different methods as additional information, can you provide your answer to the question? \n {}""".format(question)
    elif reasoning == 'ccot':
        prefix_string += """\n\n Using the answers of different methods as additional information, generate a scene graph in JSON format for the provided image and its associated question.
{}

The scene graph should include:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Obect relationships that are relevant to answering the question.

Just generate the scene graph in JSON format. Do not say extra words.""".format(question)
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
    if 'gpt' in model or 'llama' in model:
        content = completion["choices"][0]["message"]["content"]
        return {"role": "assistant", "content": content}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

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

custom_judge_prompt = '''Here are three candidate answers for the same review text, each produced by a different reasoning method.

Review Text:
{text}

Candidate 1 (Direct reasoning):
{solution1}

Candidate 2 (Aspect-graph reasoning):
{solution2}

Candidate 3 (Sub-question reasoning):
{solution3}

Compare the three candidates. Choose the one with the most consistent and well-supported reasoning.
Output ONLY a JSON object in this exact format, no extra text:
{{"Reason": "<one sentence why>", "Index": "<1, 2, or 3>"}}
'''

if __name__ == "__main__":
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

    args = parser.parse_args()    
    if getattr(args, "vllm_base_url", ""):
        args.openai_base_url = args.vllm_base_url
    if getattr(args, "vllm_api_key", ""):
        args.openai_api_key = args.vllm_api_key

    args.model_base= None
    args.model_name= None
    args.conv_mode= None
    args.sep= ","
    args.top_p= None
    
    args.system = None
    args.messages = None
    args.query = None
    
    client, args.tokenizer, args.llava, args.image_processor, args.gemini_model = MLLM_load(args)
    
    generated_description = []

    io_outputs = read_record(args.dataset, args.model, "io", split = args.split, lecture = args.lecture)
    ccot_outputs = read_record(args.dataset, args.model, "ccot", split = args.split, lecture = args.lecture)
    ddcot_outputs = read_record(args.dataset, args.model, "ddcot", split = args.split, lecture = args.lecture)
 

    if args.dataset == "custom":
        safe_model = get_safe_model_label(args.model)
        base_out = os.path.join(os.getcwd(), "outputs_none_lectures")
        # 【关键：直接写死你的实际文件名，不要用 safe_model】
        actual_model_name = "hf_unsloth_Meta-Llama-3.1-8B-bnb-4bit"
        io_path = os.path.join(base_out, f"{args.dataset}_{args.split}_{actual_model_name}_io.json")
        ccot_path = os.path.join(base_out, f"{args.dataset}_{args.split}_{actual_model_name}_ccot.json")
        ddcot_path = os.path.join(base_out, f"{args.dataset}_{args.split}_{actual_model_name}_ddcot.json")
        
# 传入 path 参数后，read_record 内部会优先使用这个 path，而不会去管 args.model 是什么
        io_outputs = read_record(args.dataset, args.model, "io", path=io_path)
        ccot_outputs = read_record(args.dataset, args.model, "ccot", path=ccot_path)
        ddcot_outputs = read_record(args.dataset, args.model, "ddcot", path=ddcot_path)
        
        # --- 调试：运行前必须看到这里打印出数字 ---
        print(f"DEBUG: Found {len(io_outputs)} IO records")
    if not os.path.exists(os.path.join(base_path, f'outputs_DMAD')):
        os.mkdir(os.path.join(base_path, f'outputs_DMAD'))
    safe_model = get_safe_model_label(args.model)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    if args.discussion_mode:
        suffix += "_discussion"
    output_path = os.path.join(base_path, f'outputs_DMAD/{args.dataset}_{args.split}_{safe_model}_{agents}_{rounds}{suffix}.json')

    print(f"[DEBUG] Output path: {output_path}")

    existing_outputs = read_json_from_path(output_path)
    
    # === 计算断点位置（start_idx） ===
    start_idx = 0
    if isinstance(existing_outputs, list) and len(existing_outputs) > 0:
        generated_description = existing_outputs   # 关键：把旧结果带入内存
        start_idx = len(existing_outputs) 
    else:
        generated_description = []
    for n in range(start_idx,len(io_outputs)):
        example = io_outputs[n]
        #question = example['question']
        if args.dataset == "ScienceQA":
            choices  = example['choices']
            hint = example['hint']
            answer   = example['choices'][example['answer']]
            question = create_prompt(question, choices, context = hint)
            args.image_path = os.path.join(base_path, f"dataset/ScienceQA/{n}.png")
        elif args.dataset == "mm-vet":
            question = create_prompt(question)
            imagename = io_outputs[n]['imagename']
            args.image_path = os.path.join(base_path, 'dataset/mm-vet/images', imagename)
        elif args.dataset == "custom":
            text = str(example["text"])  # 从 text 字段读取评论文本
            question = create_custom_prompt(text)  # 用你的任务定义包装成“question prompt”

            # 保障：把原始 ID 信息挂到 example（避免后续丢失）
            if 'raw_id' in example:
                example['_raw_id'] = example['raw_id']
            if 'num' in example:
                example['_num'] = example['num']
            args.image_path = None
            args.image_file = None
        
        example['scene_graphs'] = ccot_outputs[n]['scene_graphs'][0]
        example['subquestion_answers'] = ddcot_outputs[n]['subquestion_answerses'][0]
        example['io_output'] = io_outputs[n]['outputs'][0]
        example['ccot_output'] = ccot_outputs[n]['outputs'][0]
        example['ddcot_output'] = ddcot_outputs[n]['outputs'][0]
                  
        args.image_path = None
        args.image_file = None

   
        agent_contexts = [[question] for agent in range(agents)]
        answers_ = [[] for round in range(rounds)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round == 0:
                    # --- Round 0: 初始数据加载 ---
                    if i == 0:
                        assistant_message = 'Directly answer the question. ' + io_outputs[n][f'outputs'][0]
                        answer = io_outputs[n][f'outputs'][0]
                    elif i == 1:
                        scene_graph = ccot_outputs[n][f'scene_graphs'][0]
                        output = ccot_outputs[n][f'outputs'][0]
                        assistant_message = create_custom_ccot_answer_prompt(scene_graph = scene_graph, answer = output)
                        answer = output
                    elif i == 2:
                        subquestion_answers = ddcot_outputs[n][f'subquestion_answerses'][0]
                        output = ddcot_outputs[n][f'outputs'][0]
                        assistant_message = create_custom_ddcot_answer_prompt(subquestion_answers = subquestion_answers, answer = output)
                        answer = output
                else:
                    # --- Round 1+: 辩论逻辑 ---
                    # 确保这里能正确生成 agent_contexts_other
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    
                    if i == 0: reasoning = 'io'
                    elif i == 1: reasoning = 'ccot'
                    elif i == 2: reasoning = 'ddcot'
                    
                    # 构造消息并生成回复
                    message = construct_message(agent_contexts_other, question, 2*round - 1, 'io')
                    reasoning_message = construct_message(agent_contexts_other, question, 2*round - 1, reasoning)
                    args.messages = agent_context + [reasoning_message]
                    assistant_message = MLLM_generate(args)
                    
                    # 后处理逻辑
                    if reasoning == 'io':
                        answer = assistant_message
                        assistant_message = 'Directly answer the question. ' + assistant_message 
                    elif reasoning == 'ccot':
                        scene_graph = assistant_message
                        args.messages = agent_context + [message + f''' The scene graph of the image in JSON format:
{scene_graph}

Use the image and scene graph as context and answer the question.''']
                        answer = MLLM_generate(args)
                        assistant_message = f'''First, get the aspect graph of the image in JSON format:
{scene_graph}

Then, use the image and aspect graph as context to answer the question.
{answer}'''
                    elif reasoning == 'ddcot':
                        subquestion_answers = assistant_message
                        args.messages = agent_context + [message + f'''The problem can be deconstructed down to sub-questions.
{subquestion_answers}
According to the sub-questions and sub-answers, give your option of the problem. Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer novelty, you should answer: 
The answer is 1 otherwise 0. 
Because ...''']       
                        
                        answer = MLLM_generate(args)
                        assistant_message = f'''First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{answer}'''

                    # 【修复点 1】：将 message 的 append 缩进在这里 (else 分支内)
                    agent_context.append(message)

                # 【修复点 2】：这两行必须在 if/else 外面，但在 for i 里面
                # 这保证了 Round 0 运行完，context 长度会从 1 变成 2
                agent_context.append(assistant_message)
                answers_[round].append(answer)

            # 【修复点 3】：这个保存逻辑应该在 for i 循环结束后，但在 for round 循环内
            if args.dataset == "custom":
                example['agent_contexts'] = agent_contexts
                example['answers_'] = answers_
            
            if args.dataset == "mm-vet":
                example['eval_output'] = []
                example['eval_reason'] = []
                example['eval_index'] = []
                example['eval_solution'] = []
                example['eval_answer'] = []
                for j in range(0, rounds):
                    solutions = [agent_context[2*j+1] for agent_context in agent_contexts]
                    args.query = eval_prompt.format(question = question, solution1 = solutions[0], solution2 = solutions[1], solution3 = solutions[2])
                    args.query += json_format_prompt
                    output = MLLM_generate(args)
                    example['eval_output'].append(output)
                    try:
                        index = re.search(r"ndex\": \"(.*)\"", output).group(1)
                    except:
                        index = random.choice(['1', '2', '3'])
                    if index not in ['1', '2', '3']:
                        index = random.choice(['1', '2', '3'])
                    try:
                        reason = re.search(r"eason\": \"(.*)\"", output).group(1)
                    except:
                        reason = ''
                    example['eval_index'].append(index)
                    example['eval_reason'].append(reason)
                    example['eval_solution'].append(solutions[int(index)-1])
                    example['eval_answer'].append(answers_[j][int(index)-1])
                    
        example['agent_contexts'] = agent_contexts
        example['answers_'] = answers_
        example['usage'] = get_usage(args.model)
        # 显式带出 raw_id / num （若有）
        if '_raw_id' in example:
            example['raw_id'] = example['_raw_id']
            del example['_raw_id']
        if '_num' in example:
            example['num'] = example['_num']
            del example['_num']

        #del example["outputs"]
        generated_description.append(example)     
        if not os.path.exists(os.path.join(base_path, f'outputs_DMAD')):
            os.mkdir(os.path.join(base_path, f'outputs_DMAD'))
        with open(output_path, 'w') as f:
            json.dump(generated_description, f, indent=4)

