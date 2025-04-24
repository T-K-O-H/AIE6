<<<<<<< HEAD
---
title: RAG Implementation Notebook
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# RAG Implementation Notebook

This space contains a Jupyter notebook demonstrating a Retrieval Augmented Generation (RAG) implementation using OpenAI's API and Hugging Face models.

## Features
- PDF document processing
- Text chunking and embedding
- Vector database implementation
- RAG pipeline with context-aware responses

## How to Use
1. Clone this repository
2. Install the requirements: `pip install -r requirements.txt`
3. Open the notebook: `jupyter notebook Pythonic_RAG_Assignment.ipynb`

## Requirements
See `requirements.txt` for the complete list of dependencies.

# 🧑‍💻 What is [AI Engineering](https://maven.com/aimakerspace/ai-eng-bootcamp)?
=======
>>>>>>> 382ffca334267fe42c14491ba29014ca089abb31

<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 7: Synthetic Data Generation and LangSmith</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 7: Pre-Work](https://www.notion.so/Session-7-Synthetic-Data-Generation-for-Evaluation-1c8cd547af3d81999da6cbd18ae4b6a9?pvs=4#1c8cd547af3d81edb0f8ca8017d1cfdb)| [Session 7: Synthetic Data Generation for Evaluation](https://www.notion.so/Session-7-Synthetic-Data-Generation-for-Evaluation-1c8cd547af3d81999da6cbd18ae4b6a9) | [Recording](https://us02web.zoom.us/rec/share/fSb3DEXkbvGFQgQwwXY9Htm7CzLKXqTuOscraGRRFaj7bC7XIODwhgP2VgXpx1Uu.dKkuAcnlOrlQ7Bcq) (M9g55ye#) | [Session 7: Synthetic Data Generation for Evaluation](https://www.canva.com/design/DAGjaZbyELk/2hj3-ZHrH6x4kjz1Y3kAYw/edit?utm_content=DAGjaZbyELk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)| You Are Here!| [Session 7: Synthetic Data Generation for Evaluation](https://forms.gle/DVmFHjkkgFpAKNoEA)| [AIE6 Feedback 4/22](https://forms.gle/Z1DahTCVsNPp6SrU6)

In today's assignment, we'll be creating Synthetic Data, and using it to benchmark (and improve) a LCEL RAG Chain.

- 🤝 BREAKOUT ROOM #1
  1. Use RAGAS to Generate Synthetic Data

- 🤝 BREAKOUT ROOM #2
  1. Load them into a LangSmith Dataset
  2. Evaluate our RAG chain against the synthetic test data
  3. Make changes to our pipeline
  4. Evaluate the modified pipeline

## Ship 🚢

<<<<<<< HEAD
- Large Language Model Engineering (LLM Engineering) refers to the emerging best-practices and tools for pretraining, post-training, and optimizing LLMs prior to production deployment.  Pre- and post-training techniques include unsupervised pretraining, supervised fine-tuning, alignment, model merging, distillation, quantization. and others.
    
- Large Language Model Ops (LLM Ops, or LLMOps (as from [WandB](https://docs.wandb.ai/guides/prompts) and [a16z](https://a16z.com/emerging-architectures-for-llm-applications/))) refers to the emerging best-practices, tooling, and improvement processes used to manage production LLM applications throughout the AI product lifecycle.  LLM Ops is a subset of Machine Learning Operations (MLOps) that focuses on LLM-specific infrastructure and ops capabilities required to build, deploy, monitor, and scale complex LLM applications in production environments.  _This term is being used much less in industry these days._
=======
The completed notebook!
>>>>>>> 382ffca334267fe42c14491ba29014ca089abb31

#### 🏗️ BONUS ACTIVITY (OPTIONAL):

Reproduce the RAGAS Synthetic Data Generation Steps - but utilize a LangGraph Agent Graph, instead of the Knowledge Graph approach.

This generation should leverage the [Evol Instruct](https://arxiv.org/pdf/2304.12244) method to generate synthetic data.

Your final state (output) should contain (at least, not limited to):

1. `List(dict)`: Evolved Questions, their IDs, and their Evolution Type.
2. `List(dict)`: Question IDs, and Answer to the referenced Evolved Question.
3. `List(dict)`: Question IDs, and the relevant Context(s) to the Evolved Question.

The Graph should handle:

1. Simple Evolution.
2. Multi-Context Evolution.
3. Reasoning Evolution.

It should take, as input, a list of LangChain Documents.

### Deliverables

- A short Loom of the notebook, and a 1min. walkthrough of the application in full

## Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped Synthetic Data Generation, benchmarking, and iteration with RAGAS & LangChain! 🎉🤖

🔍 Three Key Takeaways:
1️⃣ 
2️⃣ 
3️⃣ 

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```
