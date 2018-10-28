# ASR_assignment  
Contains Code and instructions for running the ASR Assignment1

1. For Training: to train the models we have created the train.py file. We have included seperate for value of delta, number of components required for training the GMM and also whether to include the energy coefficients or not.  
The sample code for the same is:  
`train.py --components=32 --delta=1 --coefficient=Yes`  

2. For Testing there are 3 files that we have:
    1. For calculating the Frame Error Rate (**FER**): We have tie file test.py. Again this file has the following arguements:
        1. Number of components for GMM (to select the models that has been trained)
        2. Value of delta
        3. Whether you want to include the energy coefficients or not.  
    * Sample code is:  
    `test.py --components=32 --delta=1 --coefficient=Yes`  
      
      
    2. For computing the Phoneme Error Rate (**PER or WER**): First we have a python file for computing the predicted labels sentence wise and building 2 text files. Name of the file is wer_computation.py and it also has the same 3 arguements as before.  
    The code is:  
    `wer_computation.py --components=32 --delta=1 --coefficient=Yes`  
    This will generate the following two files:
        1. groundtruth.txt: that contains actual labels sentence wise.
        2. predictions.txt: that contains prected labels sentence wise.
    * After building these files, we have to pass these files to the command of asr_evaluation package and it will report the final word error rate.  
    Sample code is:  
    `wer  groundtruth.txt  predictions.txt`  
    
3. We are also reporting some of the results as follows:

| Model | No. of Components | Delta | Energy Coefficients | FER | WER |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1. | 4 | 0 | True | xx.xx | xx.xx |
| 2. | 16 | 0 | False | xx.xx | xx.xx |
| 3. | 64 | 0 | True | xx.xx | xx.xx |
| 4. | 8 | 0 | True | xx.xx | xx.xx |
| 5. | 64 | 0 | False | xx.xx | xx.xx |
| 6. | 2 | 0 | True | xx.xx | xx.xx |
| 7. | 32 | 0 | False | xx.xx | xx.xx |
| 8.| 64 | 0 | True | xx.xx | xx.xx |
