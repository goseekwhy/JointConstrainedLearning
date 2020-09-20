Repository for "Joint Constrained Learning for Event-Event Relation Extraction" (EMNLP'20)

git clone https://github.com/why2011btv/JointConstrainedLearning.git
conda env create -n conda-env -f environment.yml
pip install requirements.txt

mkdir model_params
cd model_params
mkdir HiEve_best
mkdir MATRES_best
cd ../..
nohup python3 main_aug.py gpu_0 batch_16 0.0000001 0920_0.rst epoch_40 MATRES add_loss_1 finetune_1 > output_redirect/0920_0.out 2>&1 &
nohup python3 main_aug.py gpu_1 batch_500 0.001 0920_1.rst epoch_40 MATRES add_loss_1 finetune_0 > output_redirect/0920_1.out 2>&1 &
