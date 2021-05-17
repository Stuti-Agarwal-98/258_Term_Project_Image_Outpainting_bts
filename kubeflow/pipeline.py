# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

This file defines TFX pipeline and various components in the pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text

import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImportExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component  # pylint: disable=unused-import
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input

from ml_metadata.proto import metadata_store_pb2

def create_pipeline(pipeline_name,
                   pipeline_root,
                   data_path,
                   run_fn,
                   train_args,
                   eval_args,
                   eval_accuracy_threshold,
                   serving_model_dir,
                   metadata_connection_config=None,
                   beam_pipeline_args=None,
                   ai_platform_training_args=None,
                   ai_platform_serving_args=None):
    components = []
    
    example_gen = ImportExampleGen(input_base=data_path)
    components.append(example_gen)
    
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)
        
    schema_gen = SchemaGen(
         statistics=statistics_gen.outputs['statistics'],
         infer_feature_shape=True)
    components.append(schema_gen)
    
    example_validator = ExampleValidator( 
         statistics=statistics_gen.outputs['statistics'],
         schema=schema_gen.outputs['schema'])
    components.append(example_validator)
    
    trainer_args = {
         'run_fn': run_fn,
         'examples': example_gen.outputs['examples'],
         'schema': schema_gen.outputs['schema'],
         'train_args': train_args,
         'eval_args': eval_args,
         'custom_executor_spec':
             executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
     }
    
    if ai_platform_training_args is not None:
       trainer_args.update({
           'custom_executor_spec':
               executor_spec.ExecutorClassSpec(
                   ai_platform_trainer_executor.GenericExecutor
               ),
           'custom_config': {
               ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                   ai_platform_training_args,
           }
       })
    else:
        trainer_args.update({
            'custom_config': { 'batch_size': 4,
                               'gen_epoch': 100,
                               'dis_epoch': 100,
                               'gan_epoch':100,
                               'data_size':50
                             }
        })

    trainer = Trainer(**trainer_args)
    components.append(trainer)
    
#     model_resolver = ResolverNode(
#          instance_name='latest_blessed_model_resolver',
#          resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
#          model=Channel(type=Model),
#          model_blessing=Channel(type=ModelBlessing))
#     components.append(model_resolver)
    
    eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='label')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.config.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.8})))
          ])
      ])
    
    evaluator = Evaluator(
       examples=example_gen.outputs['examples'],
       model=trainer.outputs['model'],
       eval_config=eval_config)

    pusher_args = {
         'model':
             trainer.outputs['model'],
         'model_blessing':
             evaluator.outputs['blessing'],
         'push_destination':
             pusher_pb2.PushDestination(
                 filesystem=pusher_pb2.PushDestination.Filesystem(
                     base_directory=serving_model_dir)),
    }
        
    if ai_platform_serving_args is not None:
       pusher_args.update({
           'custom_executor_spec':
               executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor
                                              ),
           'custom_config': {
               ai_platform_pusher_executor.SERVING_ARGS_KEY:
                   ai_platform_serving_args
           },
       })
    pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
    components.append(pusher)
    return pipeline.Pipeline(
         pipeline_name=pipeline_name,
         pipeline_root=pipeline_root,
         components=components,
         metadata_connection_config=metadata_connection_config,
         beam_pipeline_args=beam_pipeline_args,
     )