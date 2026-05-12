CUDA_VISIBLE_DEVICES=6  python main.py --model llama_3.1_8B --dataset custom --split test --custom_data_path novel_needs_shuffled.xlsx --openai_api_key xxx --openai_base_url http://localhost:8000/v1 --reasoning carp_json


CUDA_VISIBLE_DEVICES=6 python main.py --model llama_3.1_8B --dataset custom --split test --custom_data_path novel_needs_shuffled.xlsx --openai_api_key xxx --openai_base_url http://localhost:8000/v1 --reasoning dtg_json

CUDA_VISIBLE_DEVICES=6 python main.py --model llama_3.1_8B --dataset custom --split test --custom_data_path novel_needs_shuffled.xlsx --openai_api_key xxx --openai_base_url http://localhost:8000/v1 --max_new_tokens 2048 --reasoning dtg_negdemo


CUDA_VISIBLE_DEVICES=4 python main_MAD.py --model llama_3.1_8B --dataset custom --custom_data_path novel_needs_val.xlsx --openai_api_key xxx --openai_base_url http://localhost:8000/v1  --max_new_tokens 2048 --reasoning dtg_json 



CUDA_VISIBLE_DEVICES=5 python main.py --model llama_3.1_8B \
  --dataset custom \
  --split test \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --reasoning io_posneg \


 io_pos, io_posneg, kg, dtg_json, carp_json


CUDA_VISIBLE_DEVICES=5,6 python -m vllm.entrypoints.openai.api_server  --model unsloth/Qwen2.5-32B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --served-model-name qwen2.5-32b



CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server     --model unsloth/Meta-Llama-3.1-8B-Instruct     --served-model-name unsloth/Meta-Llama-3.1-8B-Instruct       --max-model-len 128000     --gpu-memory-utilization 0.8


#mad cot
CUDA_VISIBLE_DEVICES=3 python main_MAD.py --model llama_3.1_8B --dataset custom --custom_data_path novel_needs_val.xlsx --openai_api_key xxx --openai_base_url http://localhost:8000/v1  --max_new_tokens 2048 --reasoning cot 





#cot reasoning
CUDA_VISIBLE_DEVICES=3 python main.py --model llama_3.1_8B \
  --dataset custom \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key <你的vllm api key> \
  --openai_base_url http://localhost:8000/v1 \
  --reasoning cot

 python main.py --model qwen2.5-32b \
  --dataset custom \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key EMPTY \
  --openai_base_url http://localhost:8000/v1 \
  --max_new_tokens 2048 \
  --reasoning cot


CUDA_VISIBLE_DEVICES=6 python main.py --model llama_3.1_8B \
  --dataset custom \
  --split test \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --reasoning kg \


CUDA_VISIBLE_DEVICES=6 python main.py --model llama_3.1_8B \
  --dataset custom \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --reasoning io

 python main.py --model llama_3.1_8B \
  --dataset custom \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --reasoning sbp



  #vote
 python main_VOTE.py \
  --model llama_3.1_8B \
  --dataset custom \
  --split test \
  --agent_sources io,ccot,ddcot \
  --enable_expansion \
  --expansion_sources io2,ccot2 \
  --self_refine label_editable
  --custom_data_path novel_needs_val.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --max_new_tokens 2048

  python main_VOTE.py \
  --model llama_3.1_8B \
  --dataset custom \
  --split test \
  --agent_sources io,ccot,ddcot \
  --enable_expansion \
  --expansion_sources io2,ccot2 \
  --self_refine label_editible \
  --custom_data_path novel_needs_val.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --max_new_tokens 2048


基础版（允许 U）
export DMAD_MODEL_ID="unsloth/Meta-Llama-3.1-8B-Instruct"
  python main_MGDMAD_U.py \
  --model llama_3.1_8B \
  --dataset custom \
  --split test \
  --emit_uncertain \
  --max_debate_rounds 2 \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1



#更保守版（更强 majority anchor）
  export DMAD_MODEL_ID="unsloth/Meta-Llama-3.1-8B-Instruct"
  python main_MGDMAD_U.py \
  --model llama_3.1_8B \
  --dataset custom \
  --split test \
  --emit_uncertain \
  --strict_anchor \
  --max_debate_rounds 2
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1
  --output_suffix majority-anchor



CUDA_VISIBLE_DEVICES=1,2,3,4 python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen2.5-32B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \




  <!-- 你怎么运行？
最简单版本（先不开 MultiRole rescue）
如果你现在只想先跑一个干净的 Stage‑2 baseline，建议：
Pythonpython main_STAGE2.py \  --model qwen2.5-32b \  --dataset custom \  --split test \  --stage1_reasoning cot \  --stage1_candidate_threshold 1显示更多行
这个版本的逻辑就是：

Stage‑1：只要 5 个 CoT 输出里 至少 1 个是正类，就进入候选池。
Stage‑2：只用 primary verifier 判断 future + market。
不开 MultiRole rescue。
unsloth/Meta-Llama-3.1-8B-Instruct   
python main_STAGE2.py \
  --model unsloth/Meta-Llama-3.1-8B-Instruct \
  --dataset custom \
  --split test \
  --stage1_path ./outputs_none_lectures/custom_test_hf_unsloth_Meta-Llama-3.1-8B-bnb-4bit_io.json \
  --stage2_method cove \
  --openai_base_url http://localhost:8000/v1 \
  --openai_api_key EMPTY
  --stage1_reasoning io \


python main_STAGE2.py \
  --model qwen2.5-32b \
  --dataset custom \
  --split test \
  --stage1_path ./outputs_none_lectures/custom_test_hf_unsloth_Meta-Llama-3.1-8B-bnb-4bit_io.json \
  --stage2_method cove \
  --openai_base_url http://localhost:8000/v1 \
  --openai_api_key EMPTY \
  --stage1_reasoning io

  python main_STAGE2.py \
  --model qwen2.5-32b \
  --dataset custom \
  --split test \
  --stage1_path ./outputs_none_lectures/custom_test_hf_unsloth_Meta-Llama-3.1-8B-bnb-4bit_io.json \
  --stage2_method cove \
  --openai_base_url http://localhost:8000/v1 \
  --openai_api_key EMPTY \
  --stage2_low_votes_min 1 \
  --stage2_low_votes_max 3 \
  --stage1_high_conf_threshold 4 \
  --stage1_reasoning io \
  --output_suffix selective_cove \


# ============================================================
# RSWDV (Role-Specialized Weighted Debate Voting)
# ============================================================

# batch run (custom dataset, vLLM backend)
CUDA_VISIBLE_DEVICES=3 python main_RSWDV.py \
  --model llama_3.1_8B \
  --dataset custom \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --max_new_tokens 2048 \
  --output_suffix rswdv

# resume from checkpoint (auto-detected if output file already exists)
CUDA_VISIBLE_DEVICES=3 python main_RSWDV.py \
  --model llama_3.1_8B \
  --dataset custom \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --max_new_tokens 2048 \
  --output_path rswdv_results_rswdv.json

# single sample test
python main_RSWDV.py \
  --model llama_3.1_8B \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --text "I wish this laptop had a built-in air quality sensor."

# CoT reasoning mode (agents think step-by-step before answering)
python main_RSWDV.py \
  --model llama_3.1_8B \
  --dataset custom \
  --custom_data_path novel_needs_shuffled.xlsx \
  --openai_api_key xxx \
  --openai_base_url http://localhost:8000/v1 \
  --max_new_tokens 2048 \
  --reasoning cot \
  --output_suffix rswdv-cot
