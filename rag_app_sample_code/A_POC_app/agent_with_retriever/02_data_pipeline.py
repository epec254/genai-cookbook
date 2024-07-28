# Databricks notebook source
# MAGIC %md
# MAGIC # Unstructured data pipeline for the Agent's Retriever
# MAGIC
# MAGIC By the end of this notebook, you will have transformed your unstructured documents into a format that can be queried by your Agent.
# MAGIC
# MAGIC This means:
# MAGIC - Documents loaded into a delta table.
# MAGIC - Documents are chunked.
# MAGIC - Chunks have been embedded with an embedding model and stored in a vector index.
# MAGIC
# MAGIC The important resulting artifact of this notebook is the chunked vector index. This will be used in the next notebook to power our RAG application.
# MAGIC
# MAGIC After you have initial feedback from your stakeholders, you can easily adapt and tweak this pipeline to fit more advanced logic. For example, if your retrieval quality is low, you could experiment with different parsing and chunk sizes once you gain working knowledge of your data.
# MAGIC
# MAGIC This notebook includes smart defaults for parsing, chunking, and embedding model.  The notebook is designed such that can be easily modified based on your data e.g., customize the parsing or chunking logic, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install necessary libraries for parsing

# COMMAND ----------

# Packages required by all code
# Versions are not locked since Databricks ensures changes are backwards compatible
%pip install -qqqq -U databricks-vectorsearch databricks-agents pydantic databricks-sdk

# Versions of open source packages are locked since package authors often make backwards compatible changes

# Packages required for PDF parsing
%pip install -qqqq -U pypdf==4.1.0 

# Packages required for HTML parsing
%pip install -qqqq -U markdownify==0.12.1

# Packages required for DOCX parsing
%pip install -qqqq -U pypandoc_binary==1.13

# Packages required for get_recursive_character_text_splitter
%pip install -qqqq -U transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0

# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./utils/install_aptget_package

# COMMAND ----------

# System packages required for DOCX parsing
install_apt_get_packages(['pandoc'])

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load documents into a delta table
# MAGIC
# MAGIC In this step, we'll load files from `SOURCE_PATH` (defined in your `00_global_config`) into a delta table. The contents of each file will become a separate row in our delta table.
# MAGIC
# MAGIC The path to the source document will be used as the `doc_uri` which is displayed to your end users in the Agent's web application.
# MAGIC
# MAGIC After you test your POC with stakeholders, you can return here to change the parsing logic or extraction additional metadata about the documents to help improve the quality of your retriever.

# COMMAND ----------

# MAGIC %run ./utils/typed_dict_to_spark_schema

# COMMAND ----------

# MAGIC %run ./utils/load_uc_volume_to_delta_table

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define your parsing function
# MAGIC
# MAGIC This default implementation parses PDF, HTML, and DOCX files using open source libraries.  Adjust `file_parser(...)` to add change the parsing logic or add support for more file types.  
# MAGIC
# MAGIC

# COMMAND ----------

from typing import TypedDict
import warnings
import io
import traceback

# PDF libraries
from pypdf import PdfReader

# HTML libraries
from markdownify import markdownify as md
import markdownify
import re

## DOCX libraries
import pypandoc
import tempfile

# Schema of the dict returned by `file_parser(...)`
# WARNING: We suggest not changing this - if you need to add addition columns, we suggest doing so in the extract_metadata function in the following cells.
class ParserReturnValue(TypedDict):
    # Parsed content of the document
    doc_content: str # do not change this name
    # The status of whether the parser succeeds or fails, used to exclude failed files downstream
    parser_status: str # do not change this name

# Parser function.  Replace this function to provide custom parsing logic.
def file_parser(
    raw_doc_contents_bytes: bytes,
    doc_path: str
) -> ParserReturnValue:
    """
    Parses the content of a PDF document into a string.

    This function takes the raw bytes of a PDF document and its path, attempts to parse the document using PyPDF,
    and returns the parsed content and the status of the parsing operation. 

    Parameters:
    - raw_doc_contents_bytes (bytes): The raw bytes of the document to be parsed.
    - doc_path (str): The path of the document, used to verify the file extension.

    Returns:
    - ParserReturnValue: A dictionary containing the parsed document content and the status of the parsing operation.
      The 'doc_content' key will contain the parsed text as a string, and the 'parser_status' key will indicate
      whether the parsing was successful or if an error occurred.
    """
    try:
        file_extension = doc_path.split(".")[-1]
        if file_extension == 'pdf':
            pdf = io.BytesIO(raw_doc_contents_bytes)
            reader = PdfReader(pdf)

            parsed_content = [page_content.extract_text() for page_content in reader.pages]

            return {
                "doc_content": "\n".join(parsed_content),
                "parser_status": "SUCCESS",
            }
        elif file_extension == 'html':
            from markdownify import markdownify as md
            html_content = raw_doc_contents_bytes.decode("utf-8")

            markdown_contents = md(str(html_content).strip(), heading_style=markdownify.ATX)
            markdown_stripped = re.sub(r"\n{3,}", "\n\n", markdown_contents.strip())

            return {
                "doc_content": markdown_stripped,
                "parser_status": "SUCCESS",
            }
        elif file_extension == 'docx':
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file.write(raw_doc_contents_bytes)
                temp_file_path = temp_file.name
                md = pypandoc.convert_file(temp_file_path, "markdown", format="docx")

                return {
                    "doc_content": md.strip(),
                    "parser_status": "SUCCESS",
                }
        else:
            raise Exception(f"No supported parser for {doc_path}")
    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "doc_content": "",
            "parser_status": f"ERROR: {status}",
        }

# COMMAND ----------

md

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define your metadata extraction function
# MAGIC
# MAGIC Adjust `extract_metadata(...)` to add custom logic for extracting structured metadata from your files.  At minimum, this function must return the `doc_uri` of the document.  Add additional metadata by adding a new key to the `dict` returned by this function.
# MAGIC
# MAGIC IMPORTANT: You **MUST** add new keys to the `MetadataExtractionReturnValue` - this TypedDict used to define the schema of the Delta Table.  This pipeline will not work if the keys returned by `extract_metadata` do not match the keys in `MetadataExtractionReturnValue`.

# COMMAND ----------

from typing import TypedDict
from datetime import datetime
import warnings
import io
import traceback

# Schema of the dict returned by `extract_metadata(...)`
class MetadataExtractionReturnValue(TypedDict):
    # Unique ID of the document, required to support Agent Evaluation & Framework
    doc_uri: str # do not change this name

    # The status of whether the metadata extraction succeeds or fails, used to exclude failed files downstream
    metadata_status: str # do not change this name

    # Optionally, you can add additional metadata fields here
    example_metadata: str
    last_modified: datetime


# Replace this function to provide custom metadata extraction logic
# Do not change the signature, as the signature represents the document metadata that is extracted by the Spark pipeline that loads files from UC Volumes
def extract_metadata(
    doc_content: str, modification_time: datetime, doc_bytes_length: int, doc_path: str
) -> MetadataExtractionReturnValue:
    """
    Extracts metadata from a document.

    Parameters:
    - modification_time (timestamp): The last modification time of the document.
    - doc_bytes_length (long): The size of the document in bytes.
    - doc_content (str): The content of the document from the parsing function
    - doc_path (str): The UC Volume file path of the document.

    Returns:
    - MetadataExtractionReturnValue: A dictionary containing extracted metadata about the document.
    """

    try:
        # convert from `dbfs:/Volumes/catalog/schema/pdf_docs/filename.pdf` to `Volumes/catalog/schema/pdf_docs/filename.pdf`
        modified_path = doc_path[5:]

        # Sample metadata extraction logic
        if "test" in doc_content:
            example_metadata = "test"
        else:
            example_metadata = "not test"

        return {
            "doc_uri": modified_path,
            "last_modified": modification_time,
            "example_metadata": example_metadata,
            "metadata_status": "SUCCESS",
        }
    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "doc_uri": doc_path,
            "metadata_status": f"ERROR: {status}",
        }

# COMMAND ----------

load_uc_volume_to_delta_table(
  source_path=SOURCE_UC_VOLUME,
  dest_table_name=DOCS_DELTA_TABLE,
  # If you want to provide other URLs or document content, define these lambdas
  # to parse the URL and the content from your JSON dictionary.
  # If you need more control, change the the ./utils/json_dir_to_delta to suit your needs.
  extract_metadata_udf=extract_metadata,
  # TODO(nsthorat): Call markdownify here to show that you can do non-trivial transformations of the content.
  parse_file_udf=file_parser,
  spark_dataframe_schema=typed_dicts_to_spark_schema(ParserReturnValue, MetadataExtractionReturnValue)
)

print()
print(DOCS_DELTA_TABLE)
display(spark.table(DOCS_DELTA_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunks of documents
# MAGIC
# MAGIC Now we will split our documents into smaller chunks so they can be indexed in our vector database.
# MAGIC
# MAGIC We've provided a utility `chunk_docs` to help with this. Alternatively, you could compute chunks in your own spark pipeline and pass it to the indexing step after chunking.
# MAGIC
# MAGIC TODO(ep): Make the chunking function get the entire set of columns

# COMMAND ----------

# MAGIC %run ./utils/chunk_docs

# COMMAND ----------

# MAGIC %run ./utils/get_recursive_character_text_splitter

# COMMAND ----------

chunk_fn = get_recursive_character_text_splitter(
  # model_serving_endpoint = "databricks-bge-large-en",
  model_serving_endpoint = EMBEDDING_MODEL_ENDPOINT,
  chunk_size_tokens=1024,
  chunk_overlap_tokens=256
  )

chunked_docs_table = compute_chunks(
  # The source documents table.
  docs_table = DOCS_DELTA_TABLE,
  # The column containing the documents to be chunked.
  doc_column = 'doc_content',
  # The chunking function that takes a string (document) and returns a list of strings (chunks).
  chunk_fn = chunk_fn,
  # Choose which columns to propagate from the docs table to chunks table. `doc_uri` column is required we can propagate the original document URL to the Agent's web app.
  propagate_columns = ['doc_uri'],
  # By default, the chunked_docs_table will be written to `{docs_table}_chunked`.
  chunked_docs_table = CHUNKED_DOCS_DELTA_TABLE
)

print(chunked_docs_table)
chunked_docs_df = chunked_docs_table
display(spark.read.table(chunked_docs_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the vector index
# MAGIC
# MAGIC Next, we'll compute the vector index over the chunks and create our retriever index that will be used to query relevant documents to the user question.

# COMMAND ----------

# MAGIC %run ./utils/build_retriever_index

# COMMAND ----------

retriever_index_result = build_retriever_index(
  # Spark requires `` to escape names with special chars, VS client does not.
  chunked_docs_table = CHUNKED_DOCS_DELTA_TABLE.replace("`", ""),
  primary_key = "chunk_id",
  embedding_source_column = "content_chunked",
  vector_search_endpoint = VECTOR_SEARCH_ENDPOINT,
  vector_search_index_name = VECTOR_INDEX_NAME,
  # Must match the embedding endpoint you used to chunk your documents
  embedding_endpoint_name = EMBEDDING_MODEL_ENDPOINT,
  # Set to true to re-create the vector search endpoint when re-running.
  force_delete_vector_search_endpoint=False
)

print(retriever_index_result)

print()
print()
print('Vector search index created! This will be used in the next notebook.')
print(f'Vector search endpoint: {retriever_index_result.vector_search_endpoint}')
print(f'Vector search index: {retriever_index_result.vector_search_index_name}')
print(f'Embedding used: {retriever_index_result.embedding_endpoint_name}')
print(f'Chunked docs table: {retriever_index_result.chunked_docs_table}')

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
