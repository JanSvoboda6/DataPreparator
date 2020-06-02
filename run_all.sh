#!/bin/bash
printf "CT classification running...\n"
nohup sh -c 'python3 -u features.py ct >> SVM_ct_output.log && python3 -u classifier.py ct  >> SVM_ct_output.log' &

wait

printf "MRI BONE T1 classification running...\n"
nohup sh -c 'python3 -u features.py mri_bone_t1 >> SVM_t1_bone_output.log && python3 -u classifier.py mri_bone_t1 >> SVM_t1_bone_output.log' &

wait

printf "MRI BONE T2 classification running...\n"
nohup sh -c 'python3 -u features.py mri_bone_t2 >> SVM_t2_bone_output.log && python3 -u classifier.py mri_bone_t2 >> SVM_t2_bone_output.log' &

wait

printf "MRI TISSUE T1 classification running...\n"
nohup sh -c 'python3 -u features.py mri_tissue_t1 >> SVM_t1_tissue_output.log && python3 -u classifier.py mri_tissue_t1 >> SVM_t1_tissue_output.log' &

wait

printf "MRI TISSUE T2 classification running...\n"
nohup sh -c 'python3 -u features.py mri_tissue_t2 >> SVM_t2_tissue_output.log && python3 -u classifier.py mri_tissue_t2 >> SVM_t2_tissue_output.log' &




