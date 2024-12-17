# MLMD-news

Under review, the complete dataset will be uploaded later

# GBEGModel
## Dependency

    conda create -n ebeg python=3.9.6
    conda activate ebeg 
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    pip install -r requirements.txt

## Preprocess data
1. Put the train, test, and val files in the MLMDS dataset into the data/MLMDNews_dataset folder in the model and then run the following command:
    ```bash
    python preprocess_data.py --task train
    python preprocess_data.py --task test
    python preprocess_data.py --task val
    python ./script/lowTFIDFWords.py
    python ./script/calw2sTFIDF.py --data_path ../data/MLMDNews/train.label.jsonl
    python ./script/calw2sTFIDF.py --data_path ../data/MLMDNews/test.label.jsonl
    python ./script/calw2sTFIDF.py --data_path ../data/MLMDNews/val.label.jsonl
   ```
## Get contextualized embeddings
    python feature_extraction.py  --data ./data/MLMDNews/train.label.jsonl
    python feature_extraction.py  --data ./data/MLMDNews/test.label.jsonl
    python feature_extraction.py  --data ./data/MLMDNews/val.label.jsonl

## Training
    python train.py

## Note
To use ROUGE evaluation, you need to download the [ROUGE-1.5.5](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5) package and place it in the utils folder, then use pyrouge.<br>
**Error Handling**: If you encounter the error message Cannot open exception db file for reading: /path/to/ROUGE-1.5.5/data/WordNet-2.0.exc.db when using pyrouge, the problem can be solved from [here](https://github.com/tagucci/pythonrouge#error-handling).
