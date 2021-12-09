# 1. Prepare the Database Entries
## 1.1. Set up a MySQL server of UMLS.
Download from <https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html>.
## 1.2. Extract Entries
```bash
echo "Use umls; SELECT CUI, STR FROM MRCONSO WHERE SUPPRESS = 'N';" | mysql --user root -p --host 127.0.0.1 --port 3306 > data/umls2017aa_full.sqlout
```

# 2. Prepare the MedMentions Dataset

```bash
cd data/medmentions
bash prepare.sh # clone the dataset repository.
cd ../..
```

# 3. Prepare Environment
We provide a Dockerfile in the *docker/bio-linking* folder. To set the proper privilege, please modify the *DOCKER_UID* arg in Dockerfile before building the image.
```Dockerfile
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DOCKER_UID=1234 # <- change here
ARG DOCKER_USERNAME=experiment
ARG DOCKER_PASSWORD=passwd

RUN apt update && apt install -y less sudo
...
```

An example command to run the container is also provided on the *command/docker_enter.sh* file.

# 4. Train

```bash
python3 main.py --mode train --save_dir data/models/sample_exp --config_from configs/medmentions.json --chunk_batch_size 32
```
If your GPU doesn't have enough memory space, set a smaller value to *--chunk_batch_size*. Note that *chunk_batch_size* should be a valid divisor of the parameter *batch_size* which is set at the configuration file (*configs/medmentions.json*).

# 5. Train Again with Weight Overwriting
```bash
python3 main.py --mode train --save_dir data/models/sample_exp_second --config_from data/models/sample_exp/config.json --init_model_path data/models/sample_exp/model_best.pt --do_overwrite_weight --max_epoch 10
```

# 6. Evaluate
```bash
python3 main.py --mode eval-test --config_from data/models/sample_exp_second/config.json --init_model_path data/models/sample_exp_second/model_best.pt
```
