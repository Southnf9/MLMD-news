# MLMD-news

Under review, will upload later


## Dependency

    conda create -n ebeg python=3.9.6
    conda activate ebeg 
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    pip install requirements.txt

## Preprocess data
    python preprocess_data.py --task train
    python preprocess_data.py --task test
    python preprocess_data.py --task val
    python ./script/lowTFIDFWords.py
    python ./script/calw2sTFIDF.py --data_path ../data/MLMDNews/train.label.jsonl
    python ./script/calw2sTFIDF.py --data_path ../data/MLMDNews/test.label.jsonl
    python ./script/calw2sTFIDF.py --data_path ../data/MLMDNews/val.label.jsonl

## Get contextualized embeddings
    python feature_extraction.py  --data ./data/MLMDNews/train.label.jsonl
    python feature_extraction.py  --data ./data/MLMDNews/test.label.jsonl
    python feature_extraction.py  --data ./data/MLMDNews/val.label.jsonl

## Training
    python train.py

