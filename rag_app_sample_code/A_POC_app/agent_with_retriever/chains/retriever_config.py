# Databricks notebook source
# MAGIC %md
# MAGIC ##### RetrieverConfig
# MAGIC
# MAGIC `RetrieverConfig` is a configuration object that we use to communicate between our Agent and the Retriever tool. This configuration can be changed in conjunction with editing the Retriever code if parameters need to be changed or additional functionality added.

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

  # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question.  The f-string {chunk_text} and {metadata} can be used.
  chunk_template: str

  # Prompt template used to format all chunks for presentation to the LLM.  The f-string {context} can be used.
  prompt_template: str

  # Extra parameters to pass to DatabricksVectorSearch.as_retriever(search_kwargs=parameters).
  parameters: RetrieverParametersConfig

  # A description of the documents in the index.  Used by the Agent to determine if this tool is relevant to the query.
  description: str

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
