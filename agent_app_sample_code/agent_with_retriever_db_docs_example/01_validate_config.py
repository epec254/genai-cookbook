# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-sdk mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config to validate

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from mlflow.utils import databricks_utils as du
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist
import os
from databricks.sdk.service.compute import DataSecurityMode
from pyspark.sql import SparkSession

w = WorkspaceClient()
browser_url = du.get_browser_hostname()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if running on Single User 14.3+ cluster

# COMMAND ----------


# Get the cluster ID
spark_session = SparkSession.getActiveSession()
cluster_id = spark_session.conf.get("spark.databricks.clusterUsageTags.clusterId", None)

# # Get the current cluster name
# try:
#   cluster_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()
# except Exception as e:

cluster_info = w.clusters.get(cluster_id)

# Check if a single user cluster
# Serverless will return None here
if not cluster_info.data_security_mode == DataSecurityMode.SINGLE_USER:
  raise ValueError(f"FAIL: Current cluster is not a Single User cluster.  This notebooks currently require a single user cluster.  Please create a single user cluster: https://docs.databricks.com/en/compute/configure.html#single-node-or-multi-node-compute")

# Check for 14.3+
major_version = int(cluster_info.spark_version.split(".")[0])
minor_version = int(cluster_info.spark_version.split(".")[1])

if not ((major_version==15) or (major_version==14 and minor_version>=3)):
  raise ValueError(f"FAIL: Current cluster version {major_version}.{minor_version} is less than DBR or MLR 14.3.  Please create a DBR 14.3+ single user cluster.")
else:
  print("PASS: Running on a single user cluster version with DBR or MLR 14.3+ ({major_version}.{minor_version}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if configured locations exist
# MAGIC
# MAGIC If not, creates:
# MAGIC - UC Catalog & Schema

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
w = WorkspaceClient()

# Create UC Catalog if it does not exist, otherwise, raise an exception
try:
    _ = w.catalogs.get(UC_CATALOG)
    print(f"PASS: UC catalog `{UC_CATALOG}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=UC_CATALOG)
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
        raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
        
# Create UC Schema if it does not exist, otherwise, raise an exception
try:
    _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
    print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
        raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
