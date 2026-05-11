import json
import re
import os 
from collections import Counter
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

import os

# Set base_path to the project directory (location of this file) by default.
# Previously this was hardcoded to a Unix path which breaks on Windows.
base_path = os.path.abspath(os.path.dirname(__file__))
global safe_model

def get_safe_model_label(model: str) -> str:
    """Map a possibly long/variant model id to a short, stable filename label.

    Rules:
    - If the model name indicates Meta-Llama / Llama 3.1, return 'llama_3.1_8B'
    - Otherwise replace '/' and ':' with '_' as before.
    """
    if not model:
        return ""
    m = model.lower()
    if "meta-llama-3.1" in m or "llama-3.1" in m or "llama3.1" in m:
        return "llama_3.1_8B"
    # fallback: sanitize into filesystem-friendly name
    return model.replace("/", "_").replace(":", "_")
def record(dataset, model, reasoning, path = None, content = None, split = "test", lecture = False):
    model_label = get_safe_model_label(model)
    if path == None:
        if lecture:
            if not os.path.exists(os.path.join(base_path, "outputs_with_lectures")):
                os.mkdir(os.path.join(base_path, "outputs_with_lectures"))
            path = os.path.join(base_path,  "outputs_with_lectures", f"{dataset}_{split}_{model_label}_{reasoning}.json")
        else:
            if not os.path.exists(os.path.join(base_path, "outputs_none_lectures")):
                os.mkdir(os.path.join(base_path, "outputs_none_lectures"))
            path = os.path.join(base_path,  "outputs_none_lectures", f"{dataset}_{split}_{model_label}_{reasoning}.json")
    with open(path, "w") as f:
        json.dump(content, f, indent = 4)


def read_record(dataset, model, reasoning, path = None, split = "test", lecture = False):  
    model_label = get_safe_model_label(model)
    if path == None:
        if lecture:
            path = os.path.join(base_path, "outputs_with_lectures", f"{dataset}_{split}_{model_label}_{reasoning}.json")
        else:
            path = os.path.join(base_path, "outputs_none_lectures", f"{dataset}_{split}_{model_label}_{reasoning}.json")
    # If the unified label file doesn't exist, fall back to legacy sanitized model name
    if not os.path.exists(path):
        legacy_label = model.replace("/", "_").replace(":", "_")
        if lecture:
            fallback = os.path.join(base_path, "outputs_with_lectures", f"{dataset}_{split}_{legacy_label}_{reasoning}.json")
        else:
            fallback = os.path.join(base_path, "outputs_none_lectures", f"{dataset}_{split}_{legacy_label}_{reasoning}.json")
        if os.path.exists(fallback):
            path = fallback

    # Final fallback: scan directory for a matching dataset_split_*_{reasoning}.json
    if not os.path.exists(path):
        search_dir = os.path.join(base_path, "outputs_with_lectures" if lecture else "outputs_none_lectures")
        try:
            for fname in os.listdir(search_dir):
                if not fname.endswith(f"_{reasoning}.json"):
                    continue
                if fname.startswith(f"{dataset}_{split}_"):
                    path = os.path.join(search_dir, fname)
                    break
        except Exception:
            pass

    if os.path.exists(path):
        with open(path, "r", encoding='utf-8') as f:
            return json.loads(f.read())
    else:
        return []
        
        
def read_json_from_path(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.loads(f.read())
    else:
        return []


def write_json_from_path(path, content):
    with open(path, "w") as f:
        return json.dump(content, f, indent = 4)


def extract_answer(output = None, dataset = "ScienceQA"):
    assert output != None
    if type(output) == list:
        output = output[0]
    try:
        if dataset == "ScienceQA":
            a = re.search(r"he answer is (.)", output)
            if a != None:
                option = a.group(1)
                if option < "A" or option > "G":
                    pattern = r"\(([A-G])\)"
                    match = re.search(pattern, output)
                    
                    if match == None:
                        option = re.search(r"\*\*(.)", output).group(1)
                    else:
                        option = match.group(1)
                        if option < "A" or option > "G":
                            option = re.search(r"\*\*(.)", output).group(1)
            else:
                a = re.search(r"correct option is (.)", output)
                if a != None:
                    option = a.group(1)
                    if option < "A" or option > "G":
                        pattern = r"\(([A-G])\)"
                        match = re.search(pattern, output)
                        option = match.group(1)
                else:
                    a = re.search(r"correct answer is (.)", output)
                    if a != None:
                        option = a.group(1)
                        if option < "A" or option > "G":
                            pattern = r"\(([A-G])\)"
                            match = re.search(pattern, output)
                            option = match.group(1)
                    else:
                        pattern = r"\(([A-G])\)"
                        match = re.search(pattern, output)
                        option = match.group(1)
            #assert ord(option) >= 65 and ord(option) <= 71
    except:
        print("extract answer error!")
        print(output)
        option =  None
    return option


def find_most_common_elements(input_list):
    input_list = [input for input in input_list if input != None]
    if len(input_list) == 0:
        return [None], 0
    counter = Counter(input_list)
    max_count = max(counter.values())
    most_common_elements = [element for element, count in counter.items() if count == max_count]
    return most_common_elements, max_count
    
    
def calculate_consistency_acc(dataset, model, reasoning, lecture = None, split = "test", path = None, num = 1, agents = 3, rounds = 3):
    """
    Calculates and prints the consistency accuracy of model outputs on a given dataset using different reasoning strategies.

    Depending on the `reasoning` parameter, the function evaluates the model's answer consistency across multiple runs or rounds:
    - For "io", "ccot", or "ddcot" reasoning, it checks if the most common answer among multiple outputs matches the true answer, repeating this for each sample.
    - For "DMAD" reasoning, it evaluates accuracy across multiple rounds, considering the most common answer in each round.

    Args:
        dataset (str): The name of the dataset to evaluate.
        model (str): The model identifier.
        reasoning (str): The reasoning strategy ("io", "ccot", "ddcot", or "DMAD").
        lecture (str, optional): Additional lecture/context information. Defaults to None.
        split (str, optional): Dataset split to use ("test" by default).
        path (str, optional): Path to the output file (used for "DMAD" reasoning). Defaults to None.
        num (int, optional): Number of outputs to consider for consistency (for non-"DMAD" reasoning). Defaults to 1.
        agents (int, optional): Number of agents (used for "DMAD" reasoning). Defaults to 3.
        rounds (int, optional): Number of rounds (used for "DMAD" reasoning). Defaults to 3.

    Prints:
        Consistency accuracy or per-round accuracy for the specified reasoning strategy.
    """
    if reasoning in ["io", "ccot", "ddcot"]:
        outputs = read_record(dataset, model, split = split, lecture = lecture, reasoning = reasoning)
        right_num = 0
        length = len(outputs)
        for output in outputs:
            true_answer = output["answer"]
            answers = []
            for n in range(0, num):
                answers.append(extract_answer(output["outputs"][n]))
            most_common_elements, counts = find_most_common_elements(answers)
            answer = random.choice(most_common_elements)
            if true_answer == ord(answer) - 65:
                right_num += 1
        acc = right_num / length
        print(f"Consistency-{num} Accuracy on {dataset}: {right_num} / {length} = {acc}")
    elif reasoning == "DMAD":
        safe_model = get_safe_model_label(model)
        path = os.path.join(base_path, "outputs_DMAD", f"{dataset}_{split}_{safe_model}_{agents}_{rounds}.json")
        outputs = read_json_from_path(path)
        right_num = [0 for i in range(rounds)]
        length = len(outputs)
        for i in range(len(outputs)):
            output = outputs[i]
            true_answer = output["answer"]
            answers = []
            for n in range(rounds):
                answers = [extract_answer(answer) for answer in output["answers_"][n]]
                most_common_elements, counts = find_most_common_elements(answers)
                answer = random.choice(most_common_elements)
                if true_answer == ord(answer) - 65:
                    right_num[n] += 1
        for i in range(rounds):
            acc = right_num[i] / length 
            print(f"Accuracy of the {i}th round of DMAD on {dataset}: {right_num[i]} / {length} = {acc}")


def align_mmvet(model, reasoning, agents = 3, rounds = 3):
    path = os.path.join(base_path, "mm_vet_jsons")
    if not os.path.exists(path):
        os.mkdir(path)
    if reasoning in ["io", "ccot", "ddcot"]:
        outputs = read_record("mm-vet", model, reasoning)
        logs = {}
        for i in range(len(outputs)):
            name = f"v1_{i}"
            output = outputs[i]["outputs"][0]
            logs[name] = output
        write_json_from_path(os.path.join(path, f"{model}_{reasoning}.json"), logs)
        print(f"Results on MM-Vet are saved in {os.path.join(path, f'{model}_{reasoning}.json')}")
    elif reasoning == "DMAD":
        outputs = read_json_from_path(os.path.join(base_path, "outputs_DMAD", f"mm-vet_test_{model}_{agents}_{rounds}.json"))
        logs = [{} for i in range(rounds)]
        for i in range(len(outputs)):
            name = f"v1_{i}"
            for j in range(rounds):
                logs[j][name] = outputs[i]["eval_answer"][j]
        for j in range(rounds):
            write_json_from_path(os.path.join(path, f"{model}_{reasoning}_{agents}_{j+1}.json"), logs[j])
            print(f"Results of the {j+1}th round on MM-Vet are saved in {os.path.join(path, f'{model}_{reasoning}_{agents}_{j+1}.json')}")


# ──────────────────────────────────────────────
# Custom dataset: answer extraction & evaluation
# ──────────────────────────────────────────────

import re

# —— 预处理：统一符号 & 去装饰 & 规整空白 ——
def _normalize(s: str) -> str:
    """将常见装饰/全角符号/不可见空白规整为便于解析的形式。"""
    if not isinstance(s, str):
        s = str(s)

    # 去掉常见 Markdown/代码装饰
    s = s.replace("**", "").replace("`", "")

    # 全角冒号等替换为半角（可按需补充更多全角符号映射）
    s = s.replace("\uFF1A", ":")  # ： -> :

    # 去常见不可见空白：零宽空格/不间断空格等
    s = s.replace("\u200B", "").replace("\u00A0", " ")

    # 统一换行与多空格
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s

# —— 更宽松的 Label 抽取：接受 :, =, -, “is”，忽略大小写 ——
_LABEL_RE = re.compile(
    r"(?i)\blabel\b\s*[:=\-]?\s*(?:is\s*)?([01])\b"
)


def extract_prediction(output_text):
    """
    JSON-first parser for custom-task outputs with legacy regex fallback.

    Returns:
        {"label": int, "novel_feature": str|None, "reason": str} or None.
    """
    if output_text is None:
        return None

    text = str(output_text).strip()

    # 1) Try strict JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "label" in obj:
            label = int(obj["label"])
            if label not in (0, 1):
                return None
            return {
                "label": label,
                "novel_feature": obj.get("novel_feature"),
                "reason": obj.get("reason", obj.get("reasoning", "")),
            }
    except Exception:
        pass

    # 2) Fallback to legacy text parsing
    s = _normalize(text)
    m = _LABEL_RE.search(s)
    if m:
        return {
            "label": int(m.group(1)),
            "novel_feature": None,
            "reason": "Parsed from legacy text output",
        }

    return None

def extract_custom_label(output):
    """Extract Label: 0 or 1 from model output. Returns int or None."""
    pred = extract_prediction(output)
    return pred["label"] if pred is not None else None


def extract_custom_novel_feature(output):
    """Extract Novel Feature: ... from model output."""
    if output is None:
        return ""

    pred = extract_prediction(output)
    if pred is not None:
        nf = pred.get("novel_feature")
        if nf is None:
            return "None"
        nf_text = str(nf).strip()
        return "None" if nf_text == "" else nf_text

    s = _normalize(output)
    m = re.search(r"[Nn]ovel\s+[Ff]eature\s*:\s*(.+)", s)
    if m:
        return m.group(1).strip().split("\n")[0]
    return ""


def extract_custom_because(output):
    """Extract Because: ... from model output."""
    if output is None:
        return ""

    pred = extract_prediction(output)
    if pred is not None:
        reason = pred.get("reason", "")
        if reason is not None and str(reason).strip() != "":
            return str(reason).strip()

    s = _normalize(output)
    m = re.search(r"[Bb]ecause\s*:\s*(.+)", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""

import os
from collections import Counter
import random  # 如果你想保留随机兜底，可用；下方默认不使用

def _most_common_binary(labels):
    """输入诸如 [0,1,1,0,1]，返回多数 0/1；若平票或空，返回 None。"""
    vals = [x for x in labels if x in (0, 1)]
    if not vals:
        return None
    c = Counter(vals).most_common()
    if len(c) == 1 or c[0][1] != c[1][1]:
        return c[0][0]
    return None

def _final_label_from_answers(answers_rounds):
    """先全局多数，平票/空则回退最后一轮多数，仍无则 None。"""
    # 全局多数
    all_labels = []
    for round_answers in (answers_rounds or []):
        if not isinstance(round_answers, (list, tuple)):
            continue
        for a in round_answers:
            l = extract_custom_label(a)
            if l in (0, 1):
                all_labels.append(l)
    global_pred = _most_common_binary(all_labels)
    if global_pred in (0, 1):
        return global_pred

    # 回退到最后一轮
    if answers_rounds and isinstance(answers_rounds[-1], (list, tuple)):
        last_labels = [extract_custom_label(a) for a in answers_rounds[-1]]
        last_labels = [l for l in last_labels if l in (0, 1)]
        return _most_common_binary(last_labels)

    return None

def calculate_custom_acc(model=None, split="all", agents=3, rounds=3, path=None):
    """Evaluate custom dataset DMAD results against ground-truth label column.

    读取规则：
    - 仅使用结构化 triads_ 的最后一轮 triads_[-1][i]['y']；
    - 仅做 3 票投票（Label: 0/1），不看过程 s。

    投票规则：
    - 2 票相同或 3 票相同：多数标签获胜；
    - 无多数（平票）：随机选一个标签。
    """
    true_labels=[]
    final_preds=[]
    if path is None:
        if model is None:
            raise ValueError("model is required when path is not provided")
            safe_model = get_safe_model_label(model)
        path = os.path.join(base_path, "outputs_DMAD", f"custom_{split}_{safe_model}_{agents}_{rounds}.json")
    outputs = read_json_from_path(path)
    if not outputs:
        print(f"No output file found at {path}")
        return

    # final accuracy（最终：命中数/可判定数）
    final_right = 0
    final_total = 0

    for item in outputs:
        true_label = item.get("label")
        if true_label is None:
            continue
        try:
            true_label = int(true_label)
        except Exception:
            continue
        if true_label not in (0, 1):
            continue

        triads_rounds = item.get("triads_", [])
        if not (isinstance(triads_rounds, (list, tuple)) and len(triads_rounds) > 0):
            # 仅评估新结构化结果；无 triads_ 的旧样本直接跳过
            continue

        # 只看最后一轮的 y
        last_round = triads_rounds[-1] if len(triads_rounds) > 0 else []
        if not isinstance(last_round, (list, tuple)) or len(last_round) == 0:
            continue

        last_labels = []
        for triad in last_round:
            if isinstance(triad, dict):
                last_labels.append(extract_custom_label(triad.get("y", None)))

        valid = [l for l in last_labels if l in (0, 1)]
        counts = Counter(valid)
        if len(counts) == 0:
            final_pred = random.choice([0, 1])
        else:
            top = counts.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                final_pred = top[0][0]
            else:
                # 平票随机选一个
                final_pred = random.choice([0, 1])

        final_total += 1
        true_labels.append(true_label)
        final_preds.append(final_pred)
        if final_pred == true_label:
            final_right += 1

    # ---- 输出结果 ----
    if final_total > 0:
        final_acc = final_right / final_total
        print(f"Final (last-round 3-vote) accuracy: {final_right} / {final_total} = {final_acc:.4f}")
        
        # ---- F1, Recall, Precision ----
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, final_preds, average='binary'
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # ---- 混淆矩阵 ----
        if plt is not None and sns is not None:
            cm = confusion_matrix(true_labels, final_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['True 0', 'True 1'])
            plt.title(f"Confusion Matrix - {model}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            input_name = os.path.splitext(os.path.basename(path))[0]
            cm_path = os.path.join(base_path, "outputs_DMAD", f"confusion_matrix_{input_name}.png")
            plt.savefig(cm_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {cm_path}")
        else:
            print("Skipping confusion matrix plot: matplotlib/seaborn unavailable.")
    else:
        print("Final decision accuracy: N/A (no scorable samples)")

def align_mmvet(model, reasoning, agents = 3, rounds = 3):
    path = os.path.join(base_path, "mm_vet_jsons")
    if not os.path.exists(path):
        os.mkdir(path)
    if reasoning in ["io", "ccot", "ddcot"]:
        outputs = read_record("mm-vet", model, reasoning)
        logs = {}
        for i in range(len(outputs)):
            name = f"v1_{i}"
            output = outputs[i]["outputs"][0]
            logs[name] = output
        write_json_from_path(os.path.join(path, f"{model}_{reasoning}.json"), logs)
        print(f"Results on MM-Vet are saved in {os.path.join(path, f'{model}_{reasoning}.json')}")
    elif reasoning == "DMAD":
        outputs = read_json_from_path(os.path.join(base_path, "outputs_DMAD", f"mm-vet_test_{model}_{agents}_{rounds}.json"))
        logs = [{} for i in range(rounds)]
        for i in range(len(outputs)):
            name = f"v1_{i}"
            for j in range(rounds):
                logs[j][name] = outputs[i]["eval_answer"][j]
        for j in range(rounds):
            write_json_from_path(os.path.join(path, f"{model}_{reasoning}_{agents}_{j+1}.json"), logs[j])
            print(f"Results of the {j+1}th round on MM-Vet are saved in {os.path.join(path, f'{model}_{reasoning}_{agents}_{j+1}.json')}")
