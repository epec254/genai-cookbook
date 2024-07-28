# Databricks notebook source
# MAGIC %md
# MAGIC ##### `json_dir_to_delta`
# MAGIC
# MAGIC `json_dir_to_delta` creates a new delta table given a path to a folder in `/Volumes` which parses all the JSON files into a delta table.
# MAGIC
# MAGIC Arguments:
# MAGIC   - source_path: The path to the folder of JSON files. This can live in `/Volumes`.
# MAGIC   - dest_table_name: The destination delta table name.
# MAGIC   - get_url: A function that takes the full JSON object and returns the URL.
# MAGIC       For example: `def get_url(json): return json['url']`
# MAGIC   - get_content: A function that takes the full JSON object and returns the content.
# MAGIC       For example: `def get_content(json): return json['get_content']`
# MAGIC   - propagate_columns: A list of columns to propagate to the chunks table. This is useful for propagating context for the RAG, like the `url`.
# MAGIC   - file_extension: The file extension to glob in the output. This defaults to `json` for `.json` files.
# MAGIC

# COMMAND ----------

import json
import traceback
from typing import Any, Callable, TypedDict
import os
from IPython.display import display_markdown
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType

class ParserReturnValue(TypedDict):
  # Add more fields here if you want to add columns to the source delta table.
  content: str
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
      "content": get_content(json_dict),
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
      StructField("content", StringType(), nullable=True),
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

  num_empty_content = errors_df.filter(func.col("parsing.content") == "").count()
  if num_empty_content > 0:
    display_markdown(f"### {num_errors} documents have no content. Please review.", raw=True)
    display(errors_df)

    if num_empty_content.count() == parsed_files_staging_df.count():
      raise ValueError('All documents are empty. Please review.')

  # Filter for successfully parsed files
  parsed_files_df = (
    parsed_files_staging_df
      .filter(parsed_files_staging_df.parsing.parser_status == "SUCCESS")
      .withColumn("content", func.col("parsing.content"))
      .withColumn("url", func.col("parsing.url"))
      .drop("parsing")
  )

  # For pretty-printing the order.
  parsed_files_df = parsed_files_df.select("content", "url", "path", "modificationTime", "length")

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



# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
