# Databricks notebook source
# By default, will use the current user name to create a unique UC catalog/schema & vector search endpoint
user_email = spark.sql("SELECT current_user() as username").collect()[0].username
user_name = user_email.split("@")[0].replace(".", "").lower()[:35]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent configuration
# MAGIC
# MAGIC Important: These notebooks only work on Single User clusters running DBR/MLR 14.3+.
# MAGIC
# MAGIC To begin with, we simply need to configure the following:
# MAGIC 1. `AGENT_NAME`: The name of the Agent.  Used to name the app's Unity Catalog model and is prepended to the output Delta Tables + Vector Indexes
# MAGIC 2. `UC_CATALOG` & `UC_SCHEMA`: [Create a Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/create-catalogs.html#create-a-catalog) and a Schema where the output Delta Tables with the parsed/chunked documents and Vector Search indexes are stored
# MAGIC 3. `UC_MODEL_NAME`: Unity Catalog location to log and store the agent's model
# MAGIC
# MAGIC After finalizing your configuration, run `01_validate_config_and_create_resources` to check that you are using a valid cluster type and all locations / resources exist. Any missing resources will be created.

# COMMAND ----------

# The name of the Agent.  This is used to name the agent's UC model and prepended to the output Delta Tables + Vector Indexes
AGENT_NAME = "db_docs_app_v2"

# UC Catalog & Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f"{user_name}_catalog"
UC_SCHEMA = f"agents"

## UC Model name where the Agent's model is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional configuration
# MAGIC
# MAGIC - `MLFLOW_EXPERIMENT_NAME`: MLflow Experiment to track all experiments for this Agent.  Using the same experiment allows you to track runs across Notebooks and have unified lineage and governance for your Agent.
# MAGIC - `EVALUATION_SET_FQN`: Delta Table where your evaluation set will be stored.  In the POC, we will seed the evaluation set with feedback you collect from your stakeholders.
# MAGIC

# COMMAND ----------

############################
##### We suggest accepting these defaults unless you need to change them. ######
############################

EVALUATION_SET_FQN = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_evaluation_set`"

# MLflow experiment name
# Using the same MLflow experiment for a single app allows you to compare runs across Notebooks
MLFLOW_EXPERIMENT_NAME = f"/Users/{user_email}/{AGENT_NAME}"

# MLflow Run Names
# These Runs will store your initial POC.  They are later used to evaluate the POC model against your experiments to improve quality.

# Data pipeline MLflow run name
POC_DATA_PIPELINE_RUN_NAME = "data_pipeline_poc"
# Chain MLflow run name
POC_CHAIN_RUN_NAME = "agent_poc"

# COMMAND ----------

print('--user info--')
print(f"user_name {user_name}")

print("--agent--")
print(f"AGENT_NAME {AGENT_NAME}")
print(f"UC_CATALOG {UC_CATALOG}")
print(f"UC_SCHEMA {UC_SCHEMA}")
print(f"UC_MODEL_NAME {UC_MODEL_NAME}")

print()
print("--evaluation config--")
print(f"EVALUATION_SET_FQN {EVALUATION_SET_FQN}")
print(f"MLFLOW_EXPERIMENT_NAME {MLFLOW_EXPERIMENT_NAME}")
print(f"POC_DATA_PIPELINE_RUN_NAME {POC_DATA_PIPELINE_RUN_NAME}")
print(f"POC_CHAIN_RUN_NAME {POC_CHAIN_RUN_NAME}")

# COMMAND ----------

# # Write the parts of the configuration that are needed by the Agent model to a YAML file
# # This is used to securely provision these credentials to the deployed Agent

# agent_resources = {
#   "api_version": "1",
#   "databricks": {
#     "vector_search_index": [
#       {
#         "name": VECTOR_INDEX_NAME
#       }
#     ],
#     "serving_endpoint": [
#       {
#         "name": LLM_ENDPOINT
#       },
#       {
#         "name": EMBEDDING_MODEL_ENDPOINT
#       }
#     ]
#   }
# }

# import yaml

# # Specify the file path for the YAML file
# file_path = 'agent_resources.yaml'

# # Write the dictionary to a YAML file
# with open(file_path, 'w') as file:
#     yaml.dump(agent_resources, file)

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
