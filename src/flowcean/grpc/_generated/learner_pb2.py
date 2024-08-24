# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: learner.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rlearner.proto\"H\n\x0b\x44\x61taPackage\x12\x1b\n\x06inputs\x18\x01 \x03(\x0b\x32\x0b.TimeSeries\x12\x1c\n\x07outputs\x18\x02 \x03(\x0b\x32\x0b.TimeSeries\"N\n\nPrediction\x12 \n\x0bpredictions\x18\x01 \x03(\x0b\x32\x0b.TimeSeries\x12\x1e\n\x06status\x18\x02 \x01(\x0b\x32\x0e.StatusMessage\"*\n\nTimeSeries\x12\x1c\n\x07samples\x18\x01 \x03(\x0b\x32\x0b.TimeSample\"5\n\nTimeSample\x12\x0c\n\x04time\x18\x01 \x01(\x01\x12\x19\n\x05value\x18\x02 \x01(\x0b\x32\n.DataField\"G\n\tDataField\x12\r\n\x03int\x18\x01 \x01(\x05H\x00\x12\x10\n\x06\x64ouble\x18\x02 \x01(\x01H\x00\x12\x10\n\x06string\x18\x03 \x01(\tH\x00\x42\x07\n\x05\x66ield\"h\n\rStatusMessage\x12\x17\n\x06status\x18\x01 \x01(\x0e\x32\x07.Status\x12\x1a\n\x08messages\x18\x02 \x03(\x0b\x32\x08.Message\x12\x15\n\x08progress\x18\x03 \x01(\x05H\x00\x88\x01\x01\x42\x0b\n\t_progress\"H\n\x07Message\x12\x1c\n\tlog_level\x18\x01 \x01(\x0e\x32\t.LogLevel\x12\x0e\n\x06sender\x18\x02 \x01(\t\x12\x0f\n\x07message\x18\x03 \x01(\t\"\x07\n\x05\x45mpty*Z\n\x06Status\x12\x14\n\x10STATUS_UNDEFINED\x10\x00\x12\x12\n\x0eSTATUS_RUNNING\x10\x01\x12\x13\n\x0fSTATUS_FINISHED\x10\x02\x12\x11\n\rSTATUS_FAILED\x10\x03*\x87\x01\n\x08LogLevel\x12\x16\n\x12LOGLEVEL_UNDEFINED\x10\x00\x12\x12\n\x0eLOGLEVEL_DEBUG\x10\x01\x12\x11\n\rLOGLEVEL_INFO\x10\x02\x12\x14\n\x10LOGLEVEL_WARNING\x10\x03\x12\x12\n\x0eLOGLEVEL_ERROR\x10\x04\x12\x12\n\x0eLOGLEVEL_FATAL\x10\x05\x32x\n\x07Learner\x12)\n\x05Train\x12\x0c.DataPackage\x1a\x0e.StatusMessage\"\x00\x30\x01\x12&\n\x07Predict\x12\x0c.DataPackage\x1a\x0b.Prediction\"\x00\x12\x1a\n\x06\x45xport\x12\x06.Empty\x1a\x06.Empty\"\x00\x42\x1a\n\x18io.flowcean.learner.grpcb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'learner_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\030io.flowcean.learner.grpc'
  _globals['_STATUS']._serialized_start=532
  _globals['_STATUS']._serialized_end=622
  _globals['_LOGLEVEL']._serialized_start=625
  _globals['_LOGLEVEL']._serialized_end=760
  _globals['_DATAPACKAGE']._serialized_start=17
  _globals['_DATAPACKAGE']._serialized_end=89
  _globals['_PREDICTION']._serialized_start=91
  _globals['_PREDICTION']._serialized_end=169
  _globals['_TIMESERIES']._serialized_start=171
  _globals['_TIMESERIES']._serialized_end=213
  _globals['_TIMESAMPLE']._serialized_start=215
  _globals['_TIMESAMPLE']._serialized_end=268
  _globals['_DATAFIELD']._serialized_start=270
  _globals['_DATAFIELD']._serialized_end=341
  _globals['_STATUSMESSAGE']._serialized_start=343
  _globals['_STATUSMESSAGE']._serialized_end=447
  _globals['_MESSAGE']._serialized_start=449
  _globals['_MESSAGE']._serialized_end=521
  _globals['_EMPTY']._serialized_start=523
  _globals['_EMPTY']._serialized_end=530
  _globals['_LEARNER']._serialized_start=762
  _globals['_LEARNER']._serialized_end=882
# @@protoc_insertion_point(module_scope)
