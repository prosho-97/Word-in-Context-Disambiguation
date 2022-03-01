# Word-in-Context-Disambiguation
Comparison of three approaches for the Natural Language Processing task WiC.

## Description

An explanation regarding task and tested models can be found in the *report.pdf* file.

## OS

I developed this code on an Ubuntu 20.04.2 LTS machine.

## How to run

### Requirements

* [conda](https://docs.conda.io/projects/conda/en/latest/index.html);

* [docker](https://www.docker.com/), to avoid any issue pertaining code runnability.

  

### Notes

Unless otherwise stated, all commands here are expected to be run from the root directory of this project.

### Setup Environment

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

*test.sh* essentially setups a server exposing the model through a REST Api and then queries this server, evaluating the model. So first, you need to install Docker:

```
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding.

The model will be exposed through a REST server. In order to call it, we need a client. The client has been written
in the evaluation script, but it needs some dependencies to run. We will be using conda to create the environment for this client.

```
conda create -n nlp2021-hw1 python=3.7
conda activate nlp2021-hw1
pip install -r requirements.txt
```

### Run

*test.sh* is a simple bash script. To run it:

```
conda activate nlp2021-hw1
bash test.sh data/dev.jsonl
```

Actually, you can replace *data/dev.jsonl* to point to a different file, as far as the target file has the same format.

## Additional instructions

The *StudentModel* class takes as input two parameters. One is the *device*, the other one has as default value "SE" $\rightarrow$ the best model is tested. The other two possible parameter values are "IDF-AVG" and "SIF", that can be used in order to test the other two implemented models. In the stud folder there are the corresponding three ipython notebooks containing the three training codes.

In order to be able to run the code with the "SE" model, it is necessary to download the [glove.840B.300d](https://nlp.stanford.edu/data/glove.840B.300d.zip) vectors (they are needed also for the other two models) and the [model](https://drive.google.com/file/d/18qUSaGwkhWvkfr8l3850wFbbTZLVMtVX/view?usp=sharing) and put them in the *model/* folder.

