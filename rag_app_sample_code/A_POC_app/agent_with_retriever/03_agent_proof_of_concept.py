# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook #2: RAG proof of concept.
# MAGIC
# MAGIC By the end of this notebook, you will have created a POC of your RAG application that you can interact with, and ask questions.
# MAGIC
# MAGIC This means:
# MAGIC - We will have a mlflow model registered in the "Models" tab on the Databricks menu on the left. Models that are registered are just assets that can be instantiated from another notebook, but are not served on an endpoint. These models can be invoked with `mlflow.invoke()`.
# MAGIC - We will have a served model registered in the "Serving" tab on the Databricks menu on the left. This means that the model is served and can be accessed via a UI or a REST API for anyone in the workspace.
# MAGIC

# COMMAND ----------

# MAGIC %pip install langchain requests pyquery markdownify llama-index mlflow mlflow-skinny transformers torch==2.3.0 tiktoken==0.7.0 langchain_core==0.2.5 langchain_community==0.2.4 databricks-agents pydantic

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the MLFlow experiment
# MAGIC
# MAGIC Set the MLFlow experiment so we can log the POC.

# COMMAND ----------

import mlflow

user_email = spark.sql("SELECT current_user() as username").collect()[0].username
mlflow.set_experiment(f"/Users/{user_email}/databricks_rag")

# COMMAND ----------

# Use OpenAI client with Model Serving
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN
os.environ["DATABRICKS_HOST"] = f"{API_ROOT}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup the chain
# MAGIC
# MAGIC The `ChainConfig` here is an object which will be serialized (via json) by mlflow and passed into the configured chain notebook. These will configure the retriever as well as the LLM.
# MAGIC
# MAGIC The config here is not part of the official API, so you may change it if you see fit.

# COMMAND ----------

# MAGIC %run ./chains/chain_config

# COMMAND ----------

import os
from pprint import pprint
import yaml

# These configurations, e.g. `ChainConfig`, are not an official Databricks concept.
# They are defined in `./chains/chain_config` and are used to communicate between this notebook and the chain.
# You may edit this configuration template so that you can pass other data to the chain.

CHAIN_CODE_FILE = 'chains/single_turn_rag_chain'

##
## Retriever config.
##

retriever_config = RetrieverConfig(
  vector_search_endpoint_name=retriever_index_result.vector_search_endpoint,
  vector_search_index=retriever_index_result.vector_search_index_name,
  vector_search_schema=RetrieverSchemaConfig(
    # The primary key, the chunked text, and the document_uri column from the chunks table and vector search 
    # index above.
    primary_key="chunk_id",
    chunk_text="content_chunked",
    document_uri="url"
  ),

  # Prompt template used to format the retrieved information `chunk_text` to present to the LLM to help in answering the user's question.
  chunk_template="Passage: {chunk_text}\n",

  # Extra parameters to pass to DatabricksVectorSearch.as_retriever(search_kwargs=parameters).
  parameters=RetrieverParametersConfig(
    # The number of chunks to return for each query.
    k = 5,
    # The type of search to use, either `ann` (semantic similarity with embeddings) or `hybrid` (keyword + semantic similarity)
    query_type = "ann"
  )
)

##
## Generator config.
##

generator_config = GeneratorConfig(
  # https://docs.databricks.com/en/machine-learning/foundation-models/index.html
  llm_endpoint_name="databricks-meta-llama-3-70b-instruct",

  # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
  llm_system_prompt_template=(
"""You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.

Context: {context}"""),
  
  # Parameters that control how the LLM responds.
  llm_parameters=LLMParametersConfig(
    temperature=0.01,
    max_tokens=1500
  )
)

##
## Input example.
##
input_example = {
  "messages": [{
    "role": "user",
    "content": "What is Databricks?",
  }]
}

chain_config = ChainConfig( 
  retriever_config=retriever_config,
  generator_config=generator_config,
  input_example=input_example
)

# Write the config to to disk so we can read it from the chain.
# TODO(nsthorat): Do we actually need this?
chain_config_filepath = 'chains/generated_configs/rag_chain_config.yaml'
with open(chain_config_filepath, 'w') as f:
  yaml.dump(chain_config.dict(), f)

print(f"Using chain config:")
with open(chain_config_filepath, 'r') as f:
  print(f.read())

print(f"Using chain file: {CHAIN_CODE_FILE}")

# The chain notebook is relative to this notebook which defines the chain.
chain_notebook = os.path.join(os.getcwd(), CHAIN_CODE_FILE)

# Create the POC mlflow run.
with mlflow.start_run(run_name="poc", tags={"type": "chain"}) as mlflow_run:
  # `mlflow.langchain.log_model` will serialize the chain (defined in chains/) with the configuration
  # above so that we can execute it with `mlflow.invoke` below.
  logged_chain_info = mlflow.langchain.log_model(
      lc_model=chain_notebook,
      model_config=chain_config.dict(),
      artifact_path="chain",  # Required by MLflow
      input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
      example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
  )

  # TODO(Add the configs as explicit dictionaries we construct)

print(f"Logged MLFlow model to {logged_chain_info.model_uri}")

# COMMAND ----------


print(f'Loading MLFlow model: {logged_chain_info.model_uri}')
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
