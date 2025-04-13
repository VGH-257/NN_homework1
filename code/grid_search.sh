lrs=(1 0.1 0.01 0.001)
regs=(0 0.1 0.01 0.001)
activations=(relu sigmoid tanh)
hidden_configs=("1024 512" "512 256" "256 128" "128 64")

for lr in "${lrs[@]}"; do
  for reg in "${regs[@]}"; do
    for act in "${activations[@]}"; do
      for hidden in "${hidden_configs[@]}"; do

        hidden_str=$(echo $hidden | tr ' ' '_')
        log_file="new_log/log_lr${lr}_reg${reg}_${act}_hidden${hidden_str}.txt"
        output_path="../ckpt/best_model_lr${lr}_reg${reg}_${act}_hidden${hidden_str}.npz"

        echo "Running: lr=$lr, reg=$reg, activation=$act, hidden=$hidden"
        python train_mlp.py --lr $lr --reg $reg --activation $act --hidden $hidden  --log_path $log_file --output_path $output_path
      done
    done
  done
done