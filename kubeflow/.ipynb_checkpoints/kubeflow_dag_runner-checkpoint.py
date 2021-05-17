import os
from absl import logging
import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils
from tfx.orchestration import metadata

# Retrieve your GCP project. You can choose which project
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
 import google.auth
 try:
   _, GOOGLE_CLOUD_PROJECT = google.auth.default()
 except google.auth.exceptions.DefaultCredentialsError:
   GOOGLE_CLOUD_PROJECT = ''
except ImportError:
 GOOGLE_CLOUD_PROJECT = ''

PIPELINE_NAME = 'outpainting_pipeline_'
# Specify your GCS bucket name here. You have to use GCS to store output files
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'
RUN_FN = 'model.trainer.run_fn'
TRAIN_NUM_STEPS = 50
EVAL_NUM_STEPS = 50
EVAL_ACCURACY_THRESHOLD = 0.5
# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = os.path.join('gs://', GCS_BUCKET_NAME)
# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'advert_pred_pipeline_output',
                            PIPELINE_NAME)
# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
DATA_PATH = "./data"
print(PIPELINE_ROOT)
print(GCS_BUCKET_NAME)
METADATA_PATH = os.path.join(".", 'tfx', 'metadata',
                             'metadata.db')
def run():
     metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
     tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)
     runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
         kubeflow_metadata_config=metadata_config,
         tfx_image=tfx_image
     )

     pod_labels = kubeflow_dag_runner.get_default_pod_labels()
     pod_labels.update({telemetry_utils.LABEL_KFP_SDK_ENV: 'outpainting'})
     kubeflow_dag_runner.KubeflowDagRunner(
         config=runner_config, pod_labels_to_attach=pod_labels
     ).run(
         pipeline.create_pipeline(
             pipeline_name=PIPELINE_NAME,
             pipeline_root=PIPELINE_ROOT,
             data_path=DATA_PATH,
             run_fn=RUN_FN,
             train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_NUM_STEPS),
             eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_NUM_STEPS),
             eval_accuracy_threshold=EVAL_ACCURACY_THRESHOLD,
             serving_model_dir=SERVING_MODEL_DIR
#              metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH)
         ))
        
if __name__ == '__main__':
     logging.set_verbosity(logging.INFO)
     run()