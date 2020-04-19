#!/bin/bash
printf "==============\n"
printf "CT\n"
printf "==============\n"
nohup sh -c 'python3 -u features.py ct >> SVM_grid_ct_output.log && python3 -u classifier.py ct  >> SVM_grid_ct_output.log' &

wait

printf "==============\n"
printf "MRI BONE T1\n"
printf "==============\n"
nohup sh -c 'python3 -u features.py mri_bone_t1 >> SVM_grid_t1_bone_output.log && python3 -u classifier.py mri_bone_t1 >> SVM_grid_t1_bone_output.log' &

wait

printf "==============\n"
printf "MRI BONE T2"
printf "==============\n"
nohup sh -c 'python3 -u features.py mri_bone_t2 >> SVM_grid_t2_bone_output.log && python3 -u classifier.py mri_bone_t2 >> SVM_grid_t2_bone_output.log' &

wait

print "==============\n"
print "MRI TISSUE T1"
print "==============\n"
nohup sh -c 'python3 -u features.py mri_tissue_t1 >> SVM_grid_t1_tissue_output.log && python3 -u classifier.py mri_tissue_t1 >> SVM_grid_t1_tissue_output.log' &

wait

printf "==============\n"
printf "MRI TISSUE T2"
printf "==============\n"
nohup sh -c 'python3 -u features.py mri_tissue_t2 >> SVM_grid_t2_tissue_output.log && python3 -u classifier.py mri_tissue_t2 >> SVM_grid_t2_tissue_output.log' &




