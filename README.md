# ASR_assignment  
Contains Code and instructions for running the ASR Assignment1

1. For Training: to train the models we have created the train.py file. We have included seperate for value of delta, number of components required for training the GMM and also whether to include the energy coefficients or not.  
The sample code for the same is:  
`python train.py --components=32 --delta=1 --coefficient=Yes`  

2. For Testing there are 3 files that we have:
    1. For calculating the Frame Error Rate (**FER**): We have tie file test.py. Again this file has the following arguements:
        1. Number of components for GMM (to select the models that has been trained)
        2. Value of delta
        3. Whether you want to include the energy coefficients or not.  
    * Sample code is:  
    `python test.py --components=32 --delta=1 --coefficient=Yes`  
      
      
    2. For computing the Phoneme Error Rate (**PER or WER**): First we have a python file for computing the predicted labels sentence wise and building 2 text files. Name of the file is wer_computation.py and it also has the same 3 arguements as before.  
    The code is:  
    `python wer_computation.py --components=32 --delta=1 --coefficient=Yes`  
    This will generate the following two files:
        1. groundtruth.txt: that contains actual labels sentence wise.
        2. predictions.txt: that contains prected labels sentence wise.
    * After building these files, we have to pass these files to the command of asr_evaluation package and it will report the final word error rate.  
    Sample code is:  
    `wer  groundtruth.txt  predictions.txt `   
    `or`  
    `wer .\files_for_WER_computation\groundTruth\groundTruth_delta_0_components_4_coefficient_True.txt .\files_for_WER_computation\predicted\predict_delta_0_components_4_coefficient_True.txt
    `  
    
3. We are also reporting some of the results as follows:

| Model | No. of Components | Delta | Energy Coefficients | FER | WER |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1. | 4 | 0 | True | 86.03 | 85.138 |
| 2. | 16 | 0 | False | 86.20 | 85.315 |
| 3. | 64 | 0 | True | 85.66 | 84.999 |
| 4. | 8 | 1 | True | 83.70 | 83.048 |
| 5. | 64 | 1 | False | 82.98 | 82.385 |
| 6. | 128 | 1 | True | 81.27 | 80.736 |
| 7. | 2 | 2 | True | 85.24 | 84.369 |
| 8. | 32 | 2 | False | 83.27 | 82.656 |
| 9.| 64 | 2 | True | 81.28 | 80.760 |
| 10. | 128 | 2 | True | 81.28 | 80.75 |
| 11.|256 | 2 | False | 83.68 | 83.114 |
