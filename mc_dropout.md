# MC Dropout Score

Steps:
1) Post-training scoring phase after normal finetuning: Running the training data through the frozen model multiple times with dropout kept on to compute an uncertainty score per sample
2) Data filtering: Removal of data points from the original data set that removes the highest-uncertainty samples from the training set
3) Second training phase: Model is re-initialized and trained from scratch on the filtered dataset