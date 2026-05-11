from model import *
import model
from Dataset import *
from record import record, read_record, base_path
from reasoning import *
import argparse
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.6-13b")
    parser.add_argument("--dataset", type=str, default="ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lecture", type=bool, default=False)
    parser.add_argument("--reasoning", type=str, default="io")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--device", type=str, default = "cuda")
    parser.add_argument("--max_new_tokens", type=int, default = 2048)
    parser.add_argument("--google_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_base_url", type=str, default = "xxx")
    parser.add_argument("--openai_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_completions_url", type=str, default = "https://api.openai.com/v1/chat/completions")
    parser.add_argument("--ollama_base_url", type=str, default = "http://localhost:11434")
    parser.add_argument("--max_samples", type=int, default = -1, help="limit number of samples, -1 means all")
    parser.add_argument("--custom_data_path", type=str, default="", help="path to custom Excel/CSV file (used when --dataset custom)")
    parser.add_argument("--begin_num", type=int, default=0, help="starting sample number (for resuming partial runs)")
  
    args = parser.parse_args()    
    if getattr(args, "vllm_base_url", ""):
        args.openai_base_url = args.vllm_base_url
    if getattr(args, "vllm_api_key", ""):
        args.openai_api_key = args.vllm_api_key

    
    args = parser.parse_args()
    args.model_base= None
    args.model_name= None
    args.conv_mode= None
    args.sep= ","
    args.top_p= None
    
    args.system = None
    args.messages = None
    args.query = None
    
    # 支持 custom_data_path 为 .json 时直接读取 json
    if args.dataset == "custom" and args.reasoning == "all_io" and args.custom_data_path.endswith(".json"):
        import json
        with open(args.custom_data_path, "r") as f:
            data = json.load(f)
        print(f"[All-IO] Loaded {len(data)} samples from {args.custom_data_path}")
    else:
        data = read_dataset(args.dataset, args.split if args.dataset != "custom" else args.custom_data_path)
        if args.dataset == "ScienceQA":
            data = [d for d in data if d["image"] != None]
        if args.max_samples > 0:
            data = data[:args.max_samples]
    client, args.tokenizer, args.llava, args.image_processor, args.model_gemini = MLLM_load(args)
    
    logs = read_record(args.dataset, args.model, args.reasoning, split = args.split, lecture = args.lecture)
    begin_num = len(logs)
    num =  begin_num
    if num > 0:
        model.prompt_tokens = logs[-1]["usage"]["prompt_tokens"]
        model.completion_tokens = logs[-1]["usage"]["completion_tokens"]
    if args.dataset == "ScienceQA":
        if args.reasoning == "io":
            for i in range(begin_num, len(data)):
                example = data[i]
                example["num"] = num
                
                question = example["question"]
                choices  = example["choices"]
                
                image_path    = os.path.join(base_path, f"dataset/ScienceQA/{num}.png")
                answer   = example["choices"][example["answer"]]
                if args.lecture:
                    lecture  = example["lecture"]
                    hint     = example["hint"]
                else:
                    lecture = None
                    hint = None

                solution = example["solution"]
                example["image"] = "image"
                
                args.query = create_prompt(question, choices, lecture, hint)
                args.image_path= image_path
                args.image_file = Image.open(image_path)
                
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output) 
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning = args.reasoning, content = logs, split = args.split, lecture = args.lecture)
        elif args.reasoning == "ccot":
            for i in range(begin_num, len(data)):
                example = data[i]
                example["num"] = num
                
                question = example["question"]
                choices  = example["choices"]

                image_path    = os.path.join(base_path, f"dataset/ScienceQA/{num}.png")
                answer   = example["choices"][example["answer"]]
                if args.lecture:
                    lecture  = example["lecture"]
                    hint     = example["hint"]
                else:
                    lecture = None
                    hint = None
                solution = example["solution"]
                example["image"] = "image"
                
                options = choices
                
                create_scene_graph_prompt = create_ccot_create_scene_graph_prompt(question, choices, lecture, hint)
                args.image_path= image_path
                args.image_file = Image.open(image_path)
                
                outputs = []
                scene_graphs = []
                
                for n in range(0, 3):
                    args.query = create_scene_graph_prompt
                    scene_graph = MLLM_generate(args)
                    args.query = create_ccot_answer_with_scene_graph_prompt(scene_graph, question, options, lecture, hint)
                    output = MLLM_generate(args)
                    scene_graphs.append(scene_graph)
                    outputs.append(output) 
                example["scene_graphs"] = scene_graphs
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)

        elif args.reasoning == "ddcot":
            for i in range(begin_num, len(data)):
                example = data[i]
                example["num"] = num

                question = example["question"]
                choices  = example["choices"]
                image_path    = os.path.join(base_path, f"dataset/ScienceQA/{num}.png")
                answer   = example["choices"][example["answer"]]
                if args.lecture:
                    lecture  = example["lecture"]
                    hint     = example["hint"]
                else:
                    lecture = None
                    hint = None
                solution = example["solution"]
                example["image"] = "image"
                
                options = choices
                
                create_subquestions_prompt = create_ddcot_create_subquestions_prompt(question, choices, lecture, hint)
                args.image_path= image_path
                args.image_file = Image.open(image_path)
            
                outputs = []
                subquestion_answerses = []

                for n in range(0, 3):
                    args.query = create_subquestions_prompt
                    subquestion_answers = MLLM_generate(args)
                    args.query = create_ddcot_answer_with_subquestions_prompt(subquestion_answers, question, options, lecture, hint)
                    output_ddcot = MLLM_generate(args)
                    subquestion_answerses.append(subquestion_answers)
                    outputs.append(output_ddcot)
                example["subquestion_answerses"] = subquestion_answerses
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)

    elif args.dataset == "custom":
        if args.reasoning == "io":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_io_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "io_pos":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_io_pos_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "io_posneg":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_io_posneg_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "kg":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_kg_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "carp":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_carp_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "carp_json":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_carp_json_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "cot":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_cot_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, ):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "dtg_base":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_dtg_base_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "dtg_negdemo":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_dtg_negdemo_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "dtg_json":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.query = create_custom_dtg_json_prompt(text)
                args.image_path = None
                args.image_file = None
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output)
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning=args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "ccot":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.image_path = None
                args.image_file = None
                scene_graph_prompt = create_custom_ccot_scene_graph_prompt(text)
                outputs = []
                scene_graphs = []
                for n in range(0, 3):
                    args.query = scene_graph_prompt
                    aspect_graph = MLLM_generate(args)
                    args.query = create_custom_ccot_answer_prompt(aspect_graph, text)
                    output = MLLM_generate(args)
                    scene_graphs.append(aspect_graph)
                    outputs.append(output)
                example["scene_graphs"] = scene_graphs
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "ddcot":
            for i in range(begin_num, len(data)):
                example = dict(data[i])
                if "raw_id" not in example:
                    example["raw_id"] = example.get("num", i)
                example["num"] = num
                text = str(example["text"])
                args.image_path = None
                args.image_file = None
                subq_prompt = create_custom_ddcot_subquestions_prompt(text)
                outputs = []
                subquestion_answerses = []
                for n in range(0, 3):
                    args.query = subq_prompt
                    subquestion_answers = MLLM_generate(args)
                    args.query = create_custom_ddcot_answer_prompt(subquestion_answers, text)
                    output_ddcot = MLLM_generate(args)
                    subquestion_answerses.append(subquestion_answers)
                    outputs.append(output_ddcot)
                example["subquestion_answerses"] = subquestion_answerses
                example["outputs"] = outputs
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content=logs, split=args.split, lecture=args.lecture)

        elif args.reasoning == "ddcot":
            # Parallel custom DDCoT over batches (similar style to custom CCOT)
            from concurrent.futures import ThreadPoolExecutor
            step = 3
            output_count = 3

            def _process_one_example(base_args, text_str):
                """Run DDCoT for a single custom text, returning (subq_list, outputs_list)."""
                local_args = argparse.Namespace(**vars(base_args))
                local_args.image_path = None
                local_args.image_file = None
                subq_prompt_local = create_custom_ddcot_subquestions_prompt(text_str)
                local_subqs = []
                local_outputs = []
                for _ in range(output_count):
                    local_args.query = subq_prompt_local
                    subquestion_answers = MLLM_generate(local_args)
                    local_args.query = create_custom_ddcot_answer_prompt(subquestion_answers, text_str)
                    output_ddcot = MLLM_generate(local_args)
                    local_subqs.append(subquestion_answers)
                    local_outputs.append(output_ddcot)
                return local_subqs, local_outputs

            for i in range(begin_num, len(data), step):
                batch = [dict(data[j]) for j in range(i, min(i + step, len(data)))]
                texts = []
                for idx, example in enumerate(batch):
                    if "raw_id" not in example:
                        example["raw_id"] = example.get("num", i + idx)
                    example["num"] = num + idx
                    text = str(example["text"])
                    texts.append(text)

                batch_subqs = [None for _ in batch]
                batch_outputs = [None for _ in batch]

                with ThreadPoolExecutor(max_workers=step) as executor:
                    futures = {}
                    for idx, text in enumerate(texts):
                        args_cp = argparse.Namespace(**vars(args))
                        futures[executor.submit(_process_one_example, args_cp, text)] = idx
                    for future in futures:
                        idx = futures[future]
                        subqs, outs = future.result()
                        batch_subqs[idx] = subqs
                        batch_outputs[idx] = outs

                for idx, example in enumerate(batch):
                    example["subquestion_answerses"] = batch_subqs[idx]
                    example["outputs"] = batch_outputs[idx]
                    example["usage"] = get_usage(args.model)
                    logs.append(example)

                num += len(batch)
                record(args.dataset, args.model, args.reasoning, content=logs, split=args.split, lecture=args.lecture)

    elif args.dataset == "mm-vet":
        if args.reasoning == "io":
            for i in range(begin_num, len(data)):
                id = f"v1_{i}"
                example = data[id]
                imagename = data[id]["imagename"]
                img_path = os.path.join(base_path, "dataset/mm-vet/images", imagename)
                question = data[id]["question"]
                print(f"\n{id}")
                print(f"Image: {imagename}")
                
                example["num"] = num

                args.query = create_prompt(question, if_options = False)
                args.image_file= Image.open(img_path)
                args.image_path = img_path

                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output) 
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning = args.reasoning, content = logs, split = args.split, lecture = args.lecture)
        
        elif args.reasoning == "ccot":
            for i in range(begin_num, len(data)):
                id = f"v1_{i}"
                example = data[id]
                imagename = data[id]["imagename"]
                img_path = os.path.join(base_path, "dataset/mm-vet/images", imagename)
                question = data[id]["question"]
                print(f"\n{id}")
                print(f"Image: {imagename}")

                example["num"] = num

                create_scene_graph_prompt = create_ccot_create_scene_graph_prompt(question)
                args.image_file= Image.open(img_path)
                args.image_path = img_path

                outputs = []
                scene_graphs = []

                for n in range(0, 3):
                    args.query = create_scene_graph_prompt
                    scene_graph = MLLM_generate(args)
                    args.query = create_ccot_answer_with_scene_graph_prompt(scene_graph, question)
                    output = MLLM_generate(args)
                    scene_graphs.append(scene_graph)
                    outputs.append(output) 
                example["scene_graphs"] = scene_graphs
                example["outputs"] = outputs

                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)
            
        elif args.reasoning == "ddcot":
            for i in range(begin_num, len(data)):
                id = f"v1_{i}"
                example = data[id]
                imagename = data[id]["imagename"]
                img_path = os.path.join(base_path, "dataset/mm-vet/images", imagename)
                question = data[id]["question"]
                print(f"\n{id}")
                print(f"Image: {imagename}")
                
                example["num"] = num
                
                create_subquestions_prompt = create_ddcot_create_subquestions_prompt(question)
                args.image_file= Image.open(img_path)
                args.image_path = img_path
            
                outputs = []
                subquestion_answerses = []
                    
                for n in range(0, 3):
                    args.query = create_subquestions_prompt
                    subquestion_answers = MLLM_generate(args)
                    args.query = create_ddcot_answer_with_subquestions_prompt(subquestion_answers, question)
                    output_ddcot = MLLM_generate(args)
                    subquestion_answerses.append(subquestion_answers)
                    outputs.append(output_ddcot)
                example["subquestion_answerses"] = subquestion_answerses
                example["outputs"] = outputs

                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)