# Databricks notebook source
# MAGIC %md
# MAGIC ##### `load_uc_volume_to_delta_table`
# MAGIC
# MAGIC `load_uc_volume_to_delta_table` loads files from a specified source path into a Delta Table after parsing and extracting metadata.
# MAGIC
# MAGIC Arguments:
# MAGIC   - source_path: The path to the folder of files. This should be a valid directory path where the files are stored.
# MAGIC   - dest_table_name: The name of the destination Delta Table.
# MAGIC   - parse_file_udf: A user-defined function that takes the bytes of the file, parses it, and returns the parsed content and metadata.
# MAGIC       For example: `def parse_file(raw_doc_contents_bytes, doc_path): return {'doc_content': content, 'metadata': metadata}`
# MAGIC   - spark_dataframe_schema: The schema of the resulting Spark DataFrame after parsing and metadata extraction.

# COMMAND ----------

import json
import traceback
from datetime import datetime
from typing import Any, Callable, TypedDict, Dict
import os
from IPython.display import display_markdown
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType


def parse_and_extract(
  raw_doc_contents_bytes: bytes,
  modification_time: datetime,
  doc_bytes_length: int, 
  doc_path: str,
  # extract_metadata_udf: Callable[[[dict, Any]], str],
  parse_file_udf: Callable[[[dict, Any]], str]) -> Dict[str, Any]:
  """Parses raw bytes & extract metadata."""

  try:
    # Run the parser
    parser_output_dict = parse_file_udf(raw_doc_contents_bytes=raw_doc_contents_bytes, doc_path=doc_path, modification_time=modification_time, doc_bytes_length=doc_bytes_length)

    if parser_output_dict.get("parser_status") == 'SUCCESS':
      return parser_output_dict
    else:
      raise Exception(parser_output_dict.get("parser_status"))

  except Exception as e:
    status = f"An error occurred: {e}\n{traceback.format_exc()}"
    warnings.warn(status)
    return {
      "doc_content": "",
      "doc_uri": "",
      "parser_status": status,
    }


def get_parser_udf(
  # extract_metadata_udf: Callable[[[dict, Any]], str],
  parse_file_udf: Callable[[[dict, Any]], str],
  spark_dataframe_schema: StructType
):
  """Gets the Spark UDF which will parse the files in parallel.
  
  Arguments:
    - extract_metadata_udf: A function that takes parsed content and extracts the metadata
    - parse_file_udf: A function that takes the raw file and returns the parsed text.
    - spark_dataframe_schema: The resulting schema of the document delta table
  """
  # This UDF will load each file, parse the doc, and extract metadata.
  parser_udf = func.udf(
      lambda raw_doc_contents_bytes, modification_time, doc_bytes_length, doc_path: parse_and_extract(raw_doc_contents_bytes, modification_time, doc_bytes_length, doc_path, parse_file_udf),
      returnType=spark_dataframe_schema,
    )   
  return parser_udf

def load_uc_volume_to_delta_table(
  source_path: str,
  dest_table_name: str, 
  # extract_metadata_udf: Callable[[[dict, Any]], str],
  parse_file_udf: Callable[[[dict, Any]], str],
  spark_dataframe_schema: StructType
  # propagate_columns: list[str] = [],
  # file_extension: str = "json"
) -> str:
  if not os.path.exists(source_path):
    raise ValueError(f'{source_path} passed to `load_uc_volume_to_delta_table` does not exist.')

  # Load the raw riles
  raw_files_df = (
      spark.read.format("binaryFile")
      .option("recursiveFileLookup", "true")
      # .option("pathGlobFilter", f"*.{file_extension}")
      .load(source_path)
  )

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
  print('Running parsing & metadata extraction UDF in spark...')

  parser_udf = get_parser_udf(parse_file_udf, spark_dataframe_schema)

  # Run the parsing
  parsed_files_staging_df = raw_files_df.withColumn("parsing", parser_udf("content", "modificationTime", "length", "path")).drop("content")

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

    if num_empty_content == parsed_files_staging_df.count():
      raise ValueError('All documents are empty. Please review.')

  # Filter for successfully parsed files
  # Change the schema to the resulting schema
  resulting_fields = [field.name for field in spark_dataframe_schema.fields]

  parsed_files_df = (
    parsed_files_staging_df
      .filter(parsed_files_staging_df.parsing.parser_status == "SUCCESS")
  )

  # display(parsed_files_df)
  parsed_files_df = parsed_files_df.select(*[func.col(f"parsing.{field}").alias(field) for field in resulting_fields])

  # For pretty-printing the order.
  # parsed_files_df = parsed_files_df.select("content", "url", "path", "modificationTime", "length")

  # Write to a aDelta Table and overwrite it.
  # display(parsed_files_df)
  # print(type(parsed_files_df))
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

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
