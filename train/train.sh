### MNIST
python train.py --batch_size 64 --layers 6 --nepochs 300 --cf --cf_threshold --use_model --skip_connect 0 --lamda 60 --wre 1 --beta 0.5 --temperature 10 --beta_z 1 --threshold 0.9 --wcontras 1 --test --dataset MNIST

### SVHN
python train.py --batch_size 64 --layers 6 --nepochs 300 --cf --cf_threshold --use_model --skip_connect 0 --lamda 60 --wre 1 --beta 0.5 --temperature 2 --beta_z 1 --threshold 0.6 --wcontras 1 --test --dataset SVHN 

### CIFAR10
python train.py --batch_size 64 --layers 6 --nepochs 300 --cf --use_model --lamda 30 --wre 1 --beta 0.5 --temperature 6 --beta_z 3 --dataset CIFAR10 --threshold 0.9 --wcontras 3 --test --cf_threshold --skip_connect 0 --lr 0.0005 --latent_dim32 32

### CIFARAdd10
python train.py --batch_size 64 --layers 6 --nepochs 300 --cf --use_model --lamda 30 --wre 1 --beta 0.5 --temperature 6 --beta_z 3 --dataset CIFARAdd10 --threshold 0.5 --wcontras 3 --test --cf_threshold --skip_connect 0 --lr 0.0005 --latent_dim32 32

### CIFARAdd50
python train.py --batch_size 64 --layers 6 --nepochs 300 --cf --use_model --lamda 30 --wre 1 --beta 0.5 --temperature 6 --beta_z 3 --dataset CIFARAdd50 --threshold 0.5 --wcontras 3 --test --cf_threshold --skip_connect 0 --lr 0.0005 --latent_dim32 32




