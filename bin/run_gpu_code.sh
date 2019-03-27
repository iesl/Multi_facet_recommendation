#!/bin/bash
module load python3/current
#srun --partition=gpu --gres=gpu:1 --exclude="gpu-0-0" --cpus-per-task=2 --mem=20G  
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200
~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --lr 0.0001
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --lr 0.0001
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --update_target_emb
#~/anaconda3/bin/python src/main.py --coeff_opt max --update_target_emb --batch_size 200
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200  --lr2_divide 20 --clip 0.1
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --clip 0.1
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --lr2_divide 20
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --lr 0.1
