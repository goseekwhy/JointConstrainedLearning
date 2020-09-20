Repository for JointConstrainedLearning (EMNLP'20)

mkdir model_params
cd model_params
mkdir HiEve_best
mkdir MATRES_best
cd ../..
nohup python3 main_aug.py gpu_0 batch_16 0.0000001 0920_0.rst epoch_40 MATRES add_loss_1 finetune_1 > output_redirect/0920_0.out 2>&1 &

