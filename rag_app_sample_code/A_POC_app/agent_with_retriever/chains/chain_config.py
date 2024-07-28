# Databricks notebook source
# MAGIC %md
# MAGIC ##### ChainConfig
# MAGIC
# MAGIC `ChainConfig` is a configuration object that we use to communicate between our Agent notebook and the RAG chain that we log with mlflow, and also logged with mlflow. This configuration can be changed in conjunction with editing the chain file if parameters need to be edited.

# COMMAND ----------

# If you want to use this outside the context of the genai cookbook.
# %pip install pydantic

# COMMAND ----------

from pydantic import BaseModel
from typing import Literal, Any, List

# NOTE: These configs are created to communicate between the core notebook, and the executed chain via `mlflow.log_model`. They are pydantic models, which are thin wrappers around Python dictionaries, used for validation of the config.

class RetrieverSchemaConfig(BaseModel):
  # The column name in the retriever's response referred to the unique key
  # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
  primary_key: str
  # The column name in the retriever's response that contains the returned chunk.
  chunk_text: str
  # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
  document_uri: str
  # Additional metadata columns to present to the LLM.
  metadata_columns: List[str]

class RetrieverParametersConfig(BaseModel):
  # The number of chunks to return for each query.
  num_results: int
  # The type of search to use, either `ann` (semantic similarity with embeddings) or `hybrid`
  # (keyword + semantic similarity)
  query_type: Literal['ann', 'hybrid']

class RetrieverConfig(BaseModel):
  # Vector Search index that is created by the data pipeline
  vector_search_index: str

  vector_search_schema: RetrieverSchemaConfig

  # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question
  chunk_template: str
  # Extra parameters to pass to DatabricksVectorSearch.as_retriever(search_kwargs=parameters).
  parameters: RetrieverParametersConfig

  # A description of the documents in the index.  Used by the Agent to determine if this tool is relevant to the query.
  description: str

class LLMParametersConfig(BaseModel):
  # Parameters that control how the LLM responds.
  temperature: float
  max_tokens: int

class GeneratorConfig(BaseModel):
  # Databricks Model Serving endpoint name
  # This is the generator LLM where your LLM queries are sent.
  # Databricks foundational model endpoints can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
  llm_endpoint_name: str

  # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
  llm_system_prompt_template: str
  # Parameters that control how the LLM responds.
  llm_parameters: LLMParametersConfig

class ChainConfig(BaseModel):
  retriever_config: RetrieverConfig
  generator_config: GeneratorConfig
  input_example: Any

def validate_chain_config(config: dict) -> None:
  ChainConfig.parse_obj(config)

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
