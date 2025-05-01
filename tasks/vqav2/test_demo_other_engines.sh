#!/bin/bash

############ [1] Single Run ############
PROJECT_DIR="./"

LABEL="quick_demo"

TASK="vqav2"
DATA_FILE="$TASK/data/data.json"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"
CACHE_DIR="$TASK/cache"


# LLM="gpt-3.5-turbo" (not supported images)
# LLM="gpt-4-turbo" (not supported images)
# LLM="gpt-4" (not supported images)
# LLM="gpt-4o"
# LLM="gpt-4o-mini"
# LLM="gpt-4.1"
# LLM="gpt-4.1-nano"
# LLM="o1"
# LLM="o1-mini" (not supported images)
# LLM="o3"
# LLM="o3-mini" (not supported images)
# LLM="o4-mini"
# LLM="o1-pro"

# LLM="claude-3-haiku-20240307"
# LLM="claude-3-sonnet-20240229"
# LLM="claude-3-opus-20240229"
# LLM="claude-3-5-sonnet-20240620"
# LLM="claude-3-5-sonnet-20241022"
# LLM="claude-3-5-haiku-20241022"
# LLM="claude-3-7-sonnet-20250219"

# LLM="deepseek-chat"
# LLM="deepseek-reasoner"

# LLM="together-meta-llama/Llama-4-Scout-17B-16E-Instruct"
# LLM="together-Qwen/QwQ-32B"
LLM="together-Qwen/Qwen2-VL-72B-Instruct"

# LLM="gemini-1.5-pro" # (verified)
# LLM="gemini-1.5-flash-8b" # (verified)
# LLM="gemini-1.5-flash" # (verified)
# LLM="gemini-2.0-flash-lite" # (verified)
# LLM="gemini-2.0-flash" # (verified)
# LLM="gemini-2.5-pro-preview-03-25" # (verified)

# LLM="grok-2-vision-1212"
# LLM="grok-2-vision"
# LLM="grok-2-vision-latest"


ENABLED_TOOLS="Generalist_Solution_Generator_Tool"

INDEX=0

python solve.py \
--index $INDEX \
--task $TASK \
--data_file $DATA_FILE \
--llm_engine_name $LLM \

--root_cache_dir $CACHE_DIR \
--output_json_dir $OUT_DIR \
--output_types direct \
--enabled_tools "$ENABLED_TOOLS" \
--max_time 300
