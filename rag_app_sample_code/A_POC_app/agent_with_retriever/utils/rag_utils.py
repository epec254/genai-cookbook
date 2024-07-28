# Databricks notebook source
# MAGIC %md
# MAGIC ## Rag Utilities
# MAGIC
# MAGIC This notebook contains utils for building your RAG application. These utils can be edited if you want more control.

# COMMAND ----------

# MAGIC %pip install pydantic langchain mlflow mlflow-skinny databricks-vectorsearch transformers torch==2.3.0 tiktoken==0.7.0 langchain_core==0.2.5 langchain_community==0.2.4 databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config_utils

# COMMAND ----------

from typing import Dict, Any

def _flatten_nested_params(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, str]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_params(v, new_key, sep=sep))
        else:
          items[new_key] = v
    return items

def tag_delta_table(table_fqn, config):
    flat_config = _flatten_nested_params(config)
    sqls = []
    for key, item in flat_config.items():
        
        sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("{key.replace("/", "__")}" = "{item}")
        """)
    sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("table_source" = "rag_poc_pdf")
        """)
    for sql in sqls:
        # print(sql)
        spark.sql(sql)

# Helper function for display Delta Table URLs
def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    browser_url = du.get_browser_hostname()
    url = f"https://{browser_url}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

# COMMAND ----------


from typing import TypedDict, Dict
import io 
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from mlflow.utils import databricks_utils as du
from functools import partial
import tiktoken
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from databricks.vector_search.client import VectorSearchClient

def _build_index(
    vector_search_endpoint: str,
    chunked_docs_table_name: str,
    vectorsearch_index_name: str,
    embedding_endpoint_name: str,
    force_delete=False):

  # Get the vector search index
  vsc = VectorSearchClient(disable_notice=True)

  # Use optimizations if available
  dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
  if dbr_majorversion >= 14:
    spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

  def find_index(endpoint_name, index_name):
      all_indexes = vsc.list_indexes(name=vector_search_endpoint).get("vector_indexes", [])
      return vectorsearch_index_name in map(lambda i: i.get("name"), all_indexes)

  if find_index(endpoint_name=vector_search_endpoint, index_name=vectorsearch_index_name):
      if force_delete:
          vsc.delete_index(endpoint_name=vector_search_endpoint, index_name=vectorsearch_index_name)
          create_index = True
      else:
          create_index = False
  else:
      print('couldnt find index with', vector_search_endpoint, vectorsearch_index_name)
      create_index = True

  if create_index:
      print("Embedding docs & creating Vector Search Index, this can take 15 minutes or much longer if you have a larger number of documents.")
      print(f'Check status at: {get_table_url(vectorsearch_index_name)}')

      vsc.create_delta_sync_index_and_wait(
          endpoint_name=vector_search_endpoint,
          index_name=vectorsearch_index_name,
          primary_key="chunk_id",
          source_table_name=chunked_docs_table_name,
          pipeline_type="TRIGGERED",
          embedding_source_column="chunked_text",
          embedding_model_endpoint_name=embedding_endpoint_name
      )

#   tag_delta_table(vectorsearch_index_name, data_pipeline_config)
  mlflow.log_input(mlflow.data.load_delta(table_name=chunked_docs_table_name), context="chunked_docs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `load_json_to_delta`
# MAGIC
# MAGIC `load_json_to_delta` creates a new delta table given a path to a folder in `/Volumes` which parses all the JSON files into a delta table.
# MAGIC
# MAGIC Arguments:
# MAGIC - `source_path`: The path to the source files, for example: `/Volumes/uc_catalog/uc_schema/source_docs`.
# MAGIC

# COMMAND ----------

import json
import traceback
from typing import Any, Callable, TypedDict
import os
from IPython.display import display_markdown

class ParserReturnValue(TypedDict):
  # Add more fields here if you want to add columns to the source delta table.
  doc_content: str
  url: str

  # The status of whether the parser succeeds or fails.
  parser_status: str

def parse_bytes_json(
  raw_doc_contents_bytes: bytes,
  get_url: Callable[[[dict, Any]], str],
  get_content: Callable[[[dict, Any]], str]) -> ParserReturnValue:
  """Parses raw bytes into JSON and returns the parsed contents."""

  try:
    # Decode the raw bytes from Spark.
    json_str = raw_doc_contents_bytes.decode("utf-8")

    # Load the JSON contained in the bytes
    json_dict = json.loads(json_str)

    # Return the parsed json content back.
    return {
      "doc_content": get_content(json_dict),
      "url": get_url(json_dict),
      "parser_status": "SUCCESS",
    }

  except Exception as e:
    status = f"An error occurred: {e}\n{traceback.format_exc()}"
    warnings.warn(status)
    return {
      "json_content": "",
      "parser_status": status,
    }


def get_json_parser_udf(
  get_url: Callable[[[dict, Any]], str],
  get_content: Callable[[[dict, Any]], str]
):
  """Gets the Spark UDF which will parse the JSON files in parallel.
  
  Arguments:
    - get_url: A function that takes the JSON and returns the URL.
    - get_document: A function that takes the JSON and returns the document.
  """
  # This UDF will load the JSON from each file and parse the results.
  parser_udf = func.udf(
    lambda raw_doc_contents_bytes: parse_bytes_json(raw_doc_contents_bytes, get_url, get_content),
    returnType=StructType([
      StructField("doc_content", StringType(), nullable=True),
      StructField("url", StringType(), nullable=True),
      StructField("parser_status", StringType(), nullable=True),
    ]),
  )
  return parser_udf

def json_dir_to_delta(
  source_path: str,
  dest_table_name: str, 
  get_url: Callable[[[dict, Any]], str],
  get_content: Callable[[[dict, Any]], str],
  propagate_columns: list[str] = [],
  file_extension: str = "json"
) -> str:
  if not os.path.exists(source_path):
    raise ValueError(f'{source_path} passed to `json_dir_to_delta` does not exist.')

  # Load the raw riles
  raw_files_df = (
      spark.read.format("binaryFile")
      .option("recursiveFileLookup", "true")
      .option("pathGlobFilter", f"*.{file_extension}")
      .load(source_path)
  )

  # Save to a table
  # raw_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
      # destination_tables_config["raw_files_table_name"]
  # )

  # reload to get correct lineage in UC
  # raw_files_df = spark.read.table(destination_tables_config["raw_files_table_name"])

  # For debugging, show the list of files, but hide the binary content
  # display(raw_files_df.drop("content"))

  # Check that files were present and loaded
  if raw_files_df.count() == 0:
      display(
          f"`{source_path}` does not contain any files.  Open the volume and upload at least file."
      )
      raise Exception(f"`{source_path}` does not contain any files.")

  num_source_files = raw_files_df.count()
  print(f'Found {num_source_files} files in {source_path}.')
  count = 0
  limit = 5
  for file in raw_files_df.select("path").limit(limit).collect():
    print(file.path)
    count += 1
  if count < num_source_files:
    print(f'... and {num_source_files - limit} more.')

  print()
  print('Running JSON parsing UDF in spark...')

  # tag_delta_table(destination_tables_config["raw_files_table_name"], data_pipeline_config)

  # mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("raw_files_table_name")), context="raw_files")

  parser_udf = get_json_parser_udf(get_url, get_content)

  # Run the parsing
  parsed_files_staging_df = raw_files_df.withColumn("parsing", parser_udf("content")).drop("content")

  # Check and warn on any errors
  errors_df = parsed_files_staging_df.filter(
    func.col(f"parsing.parser_status") != "SUCCESS"
  )

  num_errors = errors_df.count()
  if num_errors > 0:
      display_markdown(f"### {num_errors} documents had parse errors. Please review.", raw=True)
      display(errors_df)

      if errors_df.count() == parsed_files_staging_df.count():
        raise ValueError('All documents produced an error during parsing. Please review.')

  num_empty_content = errors_df.filter(func.col("parsing.doc_content") == "").count()
  if num_empty_content > 0:
    display_markdown(f"### {num_errors} documents have no content. Please review.", raw=True)
    display(errors_df)

    if num_empty_content.count() == parsed_files_staging_df.count():
      raise ValueError('All documents are empty. Please review.')

  # Filter for successfully parsed files
  parsed_files_df = (
    parsed_files_staging_df
      .filter(parsed_files_staging_df.parsing.parser_status == "SUCCESS")
      .withColumn("doc_content", func.col("parsing.doc_content"))
      .withColumn("url", func.col("parsing.url"))
      .drop("parsing")
  )

  # For pretty-printing the order.
  parsed_files_df = parsed_files_df.select("doc_content", "url", "path", "modificationTime", "length")

  # Write to a aDelta Table and overwrite it.
  parsed_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(dest_table_name)

  # Reload to get correct lineage in UC.
  parsed_files_df = spark.table(dest_table_name)

  # Display for debugging
  print(f"Parsed {parsed_files_df.count()} documents.")
  # display(parsed_files_df)

  # tag_delta_table(destination_tables_config["parsed_docs_table_name"], data_pipeline_config)
  # mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("parsed_docs_table_name")), context="parsed_docs")
  return dest_table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## `chunk_docs`
# MAGIC
# MAGIC `chunk_docs` creates a new delta table, given a table of documents, computing the chunk function over each document to produce a chunked documents table. This utility will let you write the core business logic of the chunker, without dealing with the spark details. You can decide to write your own, or edit this code if it does not fit your use case.
# MAGIC
# MAGIC Arguments:
# MAGIC - `docs_table`: The fully qualified delta table name. For example: `my_catalog.my_schema.my_docs`
# MAGIC - `doc_column`: The name of the column where the documents can be found from `docs_table`. For example: `doc`.
# MAGIC - `chunk_fn`: A function that takes a document (str) and produces a list of chunks (list[str]).
# MAGIC - `propagate_columns`: Columns that should be propagated to the chunk table. For example: `url` to propagate the source URL.
# MAGIC - `chunked_docs_table`: An optional output table name for chunks. Defaults to `{docs_table}_chunked`.
# MAGIC
# MAGIC Returns:
# MAGIC The name of the chunked docs table.
# MAGIC
# MAGIC > An example `chunk_fn` using the markdown-aware node parser:
# MAGIC
# MAGIC ```py
# MAGIC from llama_index.core.node_parser import MarkdownNodeParser
# MAGIC parser = MarkdownNodeParser()
# MAGIC
# MAGIC def chunk_fn(doc: str) -> list[str]:
# MAGIC   documents = [Document(text=doc)]
# MAGIC   nodes = parser.get_nodes_from_documents(documents)
# MAGIC   return [node.get_content() for node in nodes]
# MAGIC ```

# COMMAND ----------

from typing import Literal, Optional, Any, Callable
import mlflow
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import explode
from typing import Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import tiktoken

def compute_chunks(
  docs_table: str,
  doc_column: str,
  chunk_fn: Callable[[str], list[str]],
  propagate_columns: list[str],
  chunked_docs_table: Optional[str] = None
) -> str:
  chunked_docs_table = chunked_docs_table or f'{docs_table}_chunked'

  raw_docs = spark.read.table(docs_table)

  parser_udf = func.udf(
      chunk_fn,
      returnType=ArrayType(StringType()),
  )
  chunked_array_docs = raw_docs.withColumn("chunked_text", parser_udf(doc_column)).drop(doc_column)
  chunked_docs = chunked_array_docs.select(*propagate_columns, explode("chunked_text").alias("chunked_text"))

  # Add a primary key: "chunk_id".
  chunks_with_ids = chunked_docs.withColumn(
      "chunk_id",
      func.md5(func.col("chunked_text"))
  )

  # Write to Delta Table
  chunks_with_ids.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
      chunked_docs_table
  )
  return chunked_docs_table


EMBEDDING_MODELS = {
    "databricks-gte-large-en": {
        "tokenizer": lambda: AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5"),
        "context_window": 8192,
        "type": "FMAPI",
    },
    "databricks-bge-large-en": {
        "tokenizer": lambda: AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
        "context_window": 512,
        "type": "FMAPI",
    },
    "text-embedding-ada-002": {"context_window": 8192, "tokenizer": lambda:
        tiktoken.encoding_for_model("text-embedding-ada-002")},
    "text-embedding-3-small": {"context_window": 8192, "tokenizer": lambda:
        tiktoken.encoding_for_model("text-embedding-3-small")},
    "text-embedding-3-large": {"context_window": 8192, "tokenizer": lambda:
        tiktoken.encoding_for_model("text-embedding-3-large")},
}

def get_recursive_character_text_splitter(embedding_model: str) -> Callable[[str], list[str]]:
  try:
    chunk_spec = EMBEDDING_MODELS[embedding_model]
  except KeyError:
    raise ValueError(f"Embedding `{embedding_model}` not found. Available models: {EMBEDDING_MODELS.keys()}")
  tokenizer = chunk_spec["tokenizer"]()
  # todo: detect tiktoken class and use from_tiktoken_encoder()
  splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_spec["context_window"])
  return splitter.split_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## `build_retriever_index`
# MAGIC
# MAGIC `build_retriever_index` will build the vector search index which is used by our RAG to retrieve relevant documents.
# MAGIC
# MAGIC Arguments:
# MAGIC - `chunked_docs_table`: The chunked documents table. There is expected to be a `chunked_text` column, a `chunk_id` column, and a `url` column.
# MAGIC - `vector_search_endpoint`: An optional vector search endpoint name. It not defined, defaults to the `{table_id}_vector_search`.
# MAGIC - `vector_search_index_name`: An optional index name. If not defined, defaults to `{chunked_docs_table}_index`.
# MAGIC - `embedding_endpoint_name`: An optional embedding endpoint name. Defaults to `databricks-bge-large-en`. Embedding endpoints can be found [here](https://docs.databricks.com/en/machine-learning/foundation-models/index.html#pay-per-token-foundation-model-apis). You may also choose to host your own embedding endpoint and use it here.
# MAGIC - `force_delete_vector_search_endpoint`: Setting this to true will rebuild the vector search endpoint.

# COMMAND ----------

from pydantic import BaseModel

class RetrieverIndexResult(BaseModel):
  vector_search_endpoint: str
  vector_search_index_name: str
  embedding_endpoint_name: str
  chunked_docs_table: str

def build_retriever_index(
  chunked_docs_table: str,
  vector_search_endpoint: Optional[str] = None,
  vector_search_index_name: Optional[str] = None,
  embedding_endpoint_name = 'databricks-bge-large-en',
  force_delete_vector_search_endpoint=False) -> RetrieverIndexResult:
  if not vector_search_endpoint:
    chunked_docs_table_id = chunked_docs_table.split('.')[-1]
    vector_search_endpoint = f'{chunked_docs_table_id}_vector_search'
  retriever_index_result = RetrieverIndexResult(
    # TODO(nsthorat): Is this right? Should we make a new vector search index for each chunked docs table?
    vector_search_endpoint=vector_search_endpoint,
    vector_search_index_name=vector_search_index_name or f'{chunked_docs_table}_index',
    embedding_endpoint_name=embedding_endpoint_name,
    chunked_docs_table=chunked_docs_table
  )

  # TODO: Validate the embedding_endpoint_name for a better error message.

  print('Creating vector search endpoint at', retriever_index_result.vector_search_endpoint)
  # Create the vector search endpoint.
  create_or_get_vector_search_endpoint(retriever_index_result.vector_search_endpoint)

  # Enable CDC for Vector Search Delta Sync
  spark.sql(
    f"ALTER TABLE {chunked_docs_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
  )

  print('Building embedding index...')
  # Building the index.
  _build_index(
      vector_search_endpoint=retriever_index_result.vector_search_endpoint,
      chunked_docs_table_name=chunked_docs_table,
      vectorsearch_index_name=retriever_index_result.vector_search_index_name,
      embedding_endpoint_name=retriever_index_result.embedding_endpoint_name,
      force_delete=force_delete_vector_search_endpoint)
  
  # Log to mlflow.
  chain_config = {
    "databricks_resources": {
        "vector_search_endpoint_name": retriever_index_result.vector_search_endpoint,
    },
    "retriever_config": {
        "vector_search_index": retriever_index_result.vector_search_endpoint,
        "data_pipeline_tag": "poc",
    }
  }
  mlflow.log_dict(chain_config, "chain_config.json")
  mlflow.end_run()

  return retriever_index_result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chain config

# COMMAND ----------

from pydantic import BaseModel
from typing import Literal, Optional, Any, Callable
import mlflow
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import explode

class RagApp(BaseModel):
  app_name: str
  uc_catalog: str
  uc_schema: str
  chunked_docs_table: Optional[str] = None
  docs_table: Optional[str] = None
  docs_doc_column: Optional[str] = None
  chunk_fn: Optional[Callable[[str], list[str]]] = None
  embedding_endpoint_name: str = 'databricks-bge-large-en'

  user_name: str

  chain: Literal['single_turn_rag_chain', 'multi_turn_rag_chain'] = 'single_turn_rag_chain'
  poc_chain_run_name = 'poc'

  source_path: Optional[str] = None
  mlflow_experiment_name: Optional[str] = None

  vector_search_endpoint: Optional[str] = None
  vector_search_index_name: Optional[str] = None

  logged_chain_info: Any

  def __init__(self, **kwargs: Any):
    # By default, will use the current user name to create a unique UC catalog/schema & vector search endpoint
    user_email = spark.sql("SELECT current_user() as username").collect()[0].username
    self.user_name = user_email.split("@")[0].replace(".", "")

    super().__init__(**kwargs)

    if self.chunked_docs_table and self.docs_table:
      raise ValueError('`chunked_docs_table` and `docs_table` cannot both be set.')
    if not self.chunked_docs_table and not self.docs_table:
      raise ValueError('`chunked_docs_table` or `docs_table` must be set.');

    if self.docs_table and not self.chunk_fn:
      raise ValueError('When using docs_table, please define a chunking function.')
    if self.docs_table:
      self.chunked_docs_table = f'{self.docs_table}_chunked'

    # Validate the uc_catalog / schema are valid.
    validate_schema(self.uc_catalog, self.uc_schema)

    # Validate that the source path is valid.
    if not self.source_path:
      self.source_path = f"/Volumes/{self.uc_catalog}/{self.uc_schema}/source_docs"
    validate_source_path(self.source_path)

    self.mlflow_experiment_name = f"/Users/{user_email}/{self.app_name}"
    mlflow.set_experiment(self.mlflow_experiment_name)

  def chunk_docs(self, docs_table: str, chunk_fn: Callable[[str], list[str]], select_columns: list[str], chunked_docs_table: Optional[str] = None) -> str:
    self.docs_table = docs_table
  
    raw_docs = spark.read.table(self.docs_table)

    parser_udf = func.udf(
        self.chunk_fn,
        returnType=ArrayType(StringType()),
    )
    chunked_array_docs = raw_docs.withColumn("chunked_text", parser_udf(self.docs_doc_column)).drop(self.docs_doc_column)
    chunked_docs = chunked_array_docs.select(*select_columns, explode("chunked_text").alias("chunked_text"))

    # Add a primary key: "chunk_id".
    chunks_with_ids = chunked_docs.withColumn(
        "chunk_id",
        func.md5(func.col("chunked_text"))
    )
    # Write to Delta Table
    chunks_with_ids.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        self.chunked_docs_table
    )
    return self.chunked_docs_table

  def build_index(self, vector_search_endpoint: str, force_delete=False):
    if not vector_search_endpoint:
      self.vector_search_endpoint = f'{self.user_name}_vector_search'
    self.vector_search_index_name = f"{self.chunked_docs_table}_index"
  
  def setup(self, force_delete=False):
    print('Creating vector search endpoint at', self.vector_search_endpoint)
    # Create the vector search endpoint.
    create_or_get_vector_search_endpoint(self.vector_search_endpoint)


    # Compute chunks if we need to.
    if self.docs_table:
      print('Computing chunks...')
      self.chunk_docs()

    # Enable CDC for Vector Search Delta Sync
    spark.sql(
        f"ALTER TABLE {self.chunked_docs_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    print('Building embedding index...')
    # Building the index.
    build_index(
        vector_search_endpoint=self.vector_search_endpoint,
        chunked_docs_table_name=self.chunked_docs_table,
        vectorsearch_index_name=self.vector_search_index_name,
        embedding_endpoint_name=self.embedding_endpoint_name,
        force_delete=force_delete)
    
    # Log to mlflow.
    chain_config = {
      "databricks_resources": {
          "vector_search_endpoint_name": self.vector_search_endpoint,
      },
      "retriever_config": {
          "vector_search_index": self.vector_search_endpoint,
          "data_pipeline_tag": "poc",
      }
    }
    mlflow.log_dict(chain_config, "chain_config.json")
    mlflow.end_run()

    # Setup the chain.
    rag_chain_config = _get_rag_chain_config(
      self.vector_search_endpoint, self.vector_search_index_name
    )
    data_pipeline_config = _get_data_pipeline_config(self.embedding_endpoint_name)
    with mlflow.start_run(run_name=self.poc_chain_run_name, tags={"type": "chain"}):
        self.logged_chain_info = mlflow.langchain.log_model(
            lc_model=os.path.join(
                os.getcwd(), self.chain
            ),  # Chain code file e.g., /path/to/the/chain.py
            model_config=rag_chain_config,  # Chain configuration set in 00_config
            artifact_path="chain",  # Required by MLflow
            input_example=rag_chain_config[
                "input_example"
            ],  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
            example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        )

        # Attach the data pipeline's configuration as parameters
        mlflow.log_params(_flatten_nested_params({"data_pipeline": data_pipeline_config}))

        # Attach the data pipeline configuration 
        mlflow.log_dict(data_pipeline_config, "data_pipeline_config.json")

  def invoke(self, input: dict):
    chain = mlflow.langchain.load_model(self.logged_chain_info.model_uri)
    return chain.invoke(input)
  
  def get_vector_search_endpoint(self):
    # Get the vector search index
    vsc = VectorSearchClient(disable_notice=True)
    return vsc.get_index(
      endpoint_name=self.get_vector_search_endpoint,
      index_name=self.vectorsearch_index_name)
  
  def get_vector_index_link(self) -> str:
    return get_table_url(self.vectorsearch_index_name)
  
  def get_chunked_docs_link(self) -> str:
    return get_table_url(self.chunked_docs_table)

def _get_data_pipeline_config(embedding_endpoint_name: str) -> dict:
  return {
      # Vector Search index configuration
      "vectorsearch_config": {
          # Pipeline execution mode.
          # TRIGGERED: If the pipeline uses the triggered execution mode, the system stops processing after successfully refreshing the source table in the pipeline once, ensuring the table is updated based on the data available when the update started.
          # CONTINUOUS: If the pipeline uses continuous execution, the pipeline processes new data as it arrives in the source table to keep vector index fresh.
          "pipeline_type": "TRIGGERED",
      },
      # Embedding model to use
      # Tested configurations are available in the `supported_configs/embedding_models` Notebook
      "embedding_config": {
          # Model Serving endpoint name
          "embedding_endpoint_name": "embedding_endpoint_name",
          "embedding_tokenizer": {
              # Name of the embedding model that the tokenizer recognizes
              "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
              # Name of the tokenizer, either `hugging_face` or `tiktoken`
              "tokenizer_source": "hugging_face",
          },
      },
      # Parsing and chunking configuration
      # Changing this configuration here will change in the parser or chunker logic changing, becuase these functions are hardcoded in the POC data pipeline.  However, the configuration here does impact those functions.
      # It is provided so you can copy / paste this configuration directly into the `Improve RAG quality` step and replicate the POC's data pipeline configuration
      "pipeline_config": {
          # File format of the source documents
          "file_format": "html",
          # Parser to use (must be present in `parser_library` Notebook)
          "parser": {"name": "html_to_markdown", "config": {}},
          # Chunker to use (must be present in `chunker_library` Notebook)
          "chunker": {
              "name": "langchain_recursive_char",
              "config": {
                  "chunk_size_tokens": 1024,
                  "chunk_overlap_tokens": 256,
              },
          },
      },
  }
def _get_rag_chain_config(vector_search_endpoint: str, vector_search_index_name: str) -> dict:
  return {
    "databricks_resources": {
        # Only required if using Databricks vector search
        "vector_search_endpoint_name": vector_search_endpoint,
        # Databricks Model Serving endpoint name
        # This is the generator LLM where your LLM queries are sent.
        "llm_endpoint_name": "databricks-dbrx-instruct",
    },
    "retriever_config": {
        # Vector Search index that is created by the data pipeline
        "vector_search_index": vector_search_index_name,
        "schema": {
            # The column name in the retriever's response referred to the unique key
            # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
            "primary_key": "chunk_id",
            # The column name in the retriever's response that contains the returned chunk.
            "chunk_text": "chunked_text",
            # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            "document_uri": "url",
        },
        # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question
        "chunk_template": "Passage: {chunk_text}\n",
        # The column name in the retriever's response that refers to the original document.
        "parameters": {
            # Number of search results that the retriever returns
            "k": 5,
            # Type of search to run
            # Semantic search: `ann`
            # Hybrid search (keyword + sementic search): `hybrid`
            "query_type": "ann",
        },
        # Tag for the data pipeline, allowing you to easily compare the POC results vs. future data pipeline configurations you try.
        "data_pipeline_tag": "poc",
    },
    "llm_config": {
        # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
        "llm_system_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.

Context: {context}""".strip(),
        # Parameters that control how the LLM responds.
        "llm_parameters": {"temperature": 0.01, "max_tokens": 1500},
    },
    "input_example": {
        "messages": [
            {
                "role": "user",
                "content": "How do I arpeggiate MIDI in ableton?",
            },
        ]
    },
  }

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
import os
from databricks.sdk.service.vectorsearch import EndpointType

def validate_schema(uc_catalog: str, uc_schema: str):
  w = WorkspaceClient()

  # Create UC Catalog if it does not exist, otherwise, raise an exception
  try:
      _ = w.catalogs.get(uc_catalog)
      print(f"PASS: UC catalog `{uc_catalog}` exists")
  except NotFound as e:
      print(f"`{uc_catalog}` does not exist, trying to create...")
      try:
          _ = w.catalogs.create(name=uc_catalog)
      except PermissionDenied as e:
          print(f"FAIL: `{uc_catalog}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
          raise ValueError(f"Unity Catalog `{uc_catalog}` does not exist.")
          
  # Create UC Schema if it does not exist, otherwise, raise an exception
  try:
      _ = w.schemas.get(full_name=f"{uc_catalog}.{uc_schema}")
      print(f"PASS: UC schema `{uc_catalog}.{uc_schema}` exists")
  except NotFound as e:
      print(f"`{uc_catalog}.{uc_schema}` does not exist, trying to create...")
      try:
          _ = w.schemas.create(name=uc_schema, catalog_name=uc_catalog)
          print(f"PASS: UC schema `{uc_catalog}.{uc_schema}` created")
      except PermissionDenied as e:
          print(f"FAIL: `{uc_catalog}.{uc_schema}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
          raise ValueError("Unity Catalog Schema `{uc_catalog}.{uc_schema}` does not exist.")

def validate_source_path(source_path: str):
  if os.path.isdir(source_path):
      print(f"PASS: `{source_path}` exists")
  else:
      print(f"`{source_path}` does NOT exist, trying to create")

      from databricks.sdk import WorkspaceClient
      from databricks.sdk.service import catalog
      from databricks.sdk.errors import ResourceAlreadyExists

      w = WorkspaceClient()

      volume_name = source_path[9:].split('/')[2]
      uc_catalog = source_path[9:].split('/')[0]
      uc_schema = source_path[9:].split('/')[1]
      try:
          created_volume = w.volumes.create(
              catalog_name=uc_catalog,
              schema_name=uc_schema,
              name=volume_name,
              volume_type=catalog.VolumeType.MANAGED,
          )
          print(f"PASS: Created `{source_path}`")
      except Exception as e:
          print(f"`FAIL: {source_path}` does NOT exist, could not create due to {e}")
          raise ValueError("Please verify that `{source_path}` is a valid UC Volume")


def create_or_get_vector_search_endpoint(vector_search_endpoint: str):
  w = WorkspaceClient()
  vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
  if sum([vector_search_endpoint == ve.name for ve in vector_search_endpoints]) == 0:
      print(f"Please wait, creating Vector Search endpoint `{vector_search_endpoint}`.  This can take up to 20 minutes...")
      w.vector_search_endpoints.create_endpoint_and_wait(vector_search_endpoint, endpoint_type=EndpointType.STANDARD)

  # Make sure vector search endpoint is online and ready.
  w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(vector_search_endpoint)

  print(f"PASS: Vector Search endpoint `{vector_search_endpoint}` exists")
