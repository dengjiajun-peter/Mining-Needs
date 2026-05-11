import os
import time
import traceback
from openai import OpenAI

# 模型名称映射（已移除默认硬编码映射，直接使用传入的模型名）
# 若需要自定义映射，可在此处添加或通过环境变量覆盖。
MODEL_NAME_MAP = {}

completion_tokens = 0
prompt_tokens = 0

def get_usage(model):
    global completion_tokens, prompt_tokens
    cost = 0
    total_tokens = completion_tokens + prompt_tokens
    return {
        "completion_tokens": completion_tokens, 
        "prompt_tokens": prompt_tokens, 
        "total_tokens": total_tokens, 
        "cost": cost
    }

def MLLM_load(args):
    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url=args.openai_base_url,
        api_key=args.openai_api_key
    )
    # 保持原有接口返回 5 个变量
    return client, None, None, None, None

def MLLM_generate(args, client=None):
    global completion_tokens, prompt_tokens
    
    # 【修正 1】：绝对不要覆盖 client 变量
    # 如果调用时没传 client，再根据 args 动态创建一个
    if client is None or isinstance(client, str):
        # 如果 client 意外变成了字符串，重新初始化
        from openai import OpenAI
        actual_client = OpenAI(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key
        )
    else:
        actual_client = client

    # 优先使用环境变量 `DMAD_MODEL_ID`（方便临时覆盖），其次使用 MODEL_NAME_MAP 映射，最后回退到 args.model_name
    model_id = os.getenv("DMAD_MODEL_ID") or MODEL_NAME_MAP.get(args.model_name, args.model_name)

    # 准备消息格式
    if hasattr(args, 'messages') and args.messages is not None:
        messages = [{"role": "user", "content": m} for m in args.messages]
    else:
        messages = [{"role": "user", "content": args.query}]

    max_retries = 3
    timeout = 120  # 建议增加超时时间
    #DMAD Model setting
    for attempt in range(max_retries):
        try:
            # 使用 actual_client 而不是被覆盖的字符串
            completion = actual_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                top_p=1.0,
                presence_penalty=1.0,
                timeout=timeout
    #Debate or vevote model setting
    # for attempt in range(max_retries):
    #     try:
    #         # 使用 actual_client 而不是被覆盖的字符串
    #         completion = actual_client.chat.completions.create(
    #             model=model_id,
    #             messages=messages,
    #             temperature=1,
    #             max_tokens=args.max_new_tokens,
    #             top_p=0.9,
    #             timeout=timeout
            )
            
            # 更新 Token 统计
            try:
                prompt_tokens += completion.usage.prompt_tokens
                completion_tokens += completion.usage.completion_tokens
            except Exception:
                pass
                
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"[MLLM_generate] API call failed (attempt {attempt+1}/{max_retries}): {e}")
            # 如果是序列化错误，打印详细堆栈
            if "not JSON serializable" in str(e):
                 traceback.print_exc()
            
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("[MLLM_generate] Max retries reached.")
                return "[ERROR: API call failed]"