python inference.py \
--validation_data_file "data/val_with_token.json" \
--model_name "meta-llama/Llama-2-7b-hf" \
--adapter_path "checkpoints/llama2_lora/checkpoint-70000-llama2" \
--results_file "output/llama2_lora/output_llama2_lora.json" \
