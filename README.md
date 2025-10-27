# TASKS

## Introduction
This repository contains the code used in our paper: **TAPE: A Temporal Graph-based Memory System for Personal LLM Agents**. TAPE is a temporal graph-based memory system designed to serve as the memory backbone for personal LLM agents. 

The demo link for TAPE is https://youtu.be/ndmagVTMzDw

## Running Environment

a 64-bit Linux-based OS; a MacOS

## Usage
#### How to run

- Step 1: Push the demo-backend part code to the linux server

- Step 2: Move to directory `/back-end` and run the python file, through which the data interaction between the front-end interface and the back-end TAPE is realized

  ```cmd
  python app.py
  ```

- Step 3: Open the front-end interface `/front-end/templates/index.html` at MacOS or Windows  to use TAPE


## Requirements
+ cmake
+ g++
+ OpenMP
+ Boost
+ flask
+ openai
+ tiktoken
+ torch==2.5.1+cu121
+ faiss-cpu==1.10.0
+ pyg_lib

