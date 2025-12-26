model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=6

master_port=29500
num_process=1
batch_size=32
d_model=32
d_ff=256

comment='TimeLLM-build-classification'


accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_build_pred.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/data/ \
  --data_path test_fail.csv \
  --model_id test_fail_ten\
  --model $model_name \
  --data build_data \
  --features S \
  --target now_label \
  --loss BCE \
  --seq_len 32 \
  --label_len 5 \
  --pred_len 1 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1 


:<<'COMMINT'

accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_build_pred_non_json.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/build_fail/ \
  --data_path cloudify.csv \
  --model_id cloudify_five_reprog_prompt\
  --model $model_name \
  --data build \
  --features S \
  --target build_Failed \
  --loss BCE \
  --seq_len 32 \
  --label_len 5 \
  --pred_len 1 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1 


accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_build_pred.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/data/ \
  --data_path test_fail.csv \
  --model_id test_fail_second\
  --model $model_name \
  --data build_data \
  --features S \
  --target now_label \
  --loss BCE \
  --seq_len 32 \
  --label_len 5 \
  --pred_len 1 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --prompt_domain 1 




accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_build_pred.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/build_fail/ \
  --data_path open-build-service.csv \
  --model_id open-build-service_seq_16_f1_2 \
  --model $model_name \
  --data build \
  --features S \
  --target build_Failed \
  --loss BCE \
  --seq_len 32 \
  --label_len 5 \
  --pred_len 1 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment  

  accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_build_pred.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/build_fail/ \
  --data_path openproject.csv \
  --model_id openproject_seq_16_f1_2 \
  --model $model_name \
  --data build \
  --features S \
  --target build_Failed \
  --loss BCE \
  --seq_len 16 \
  --label_len 4 \
  --pred_len 1 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment  

COMMINT




