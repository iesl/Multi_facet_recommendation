#!/bin/bash
module load python3/current
#srun --partition=gpu --gres=gpu:1 --exclude="gpu-0-0" --cpus-per-task=2 --mem=20G  
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/Wacky-20190329-100602 --outf ./gen_log/val_glove_lc_elayer2_bsz200_ep5_linear --nlayers 2
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/Wacky-20190329-154753 --outf ./gen_log/val_updated_glove_lc_elayer2_bsz200_ep4_linear --nlayers 2
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/book-20190331-112352 --outf ./gen_log/val_glove_book_maxlc_elayer2_bsz200_ep2_linear --nlayers 2 --data ./data/processed/bookp1/
~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org_lemma --checkpoint ./models/Wacky-20190329-100602 --outf ./gen_log/SWCS_glove_lc_elayer2_bsz200_ep5_linear.json --nlayers 2 --max_sent_len 200
