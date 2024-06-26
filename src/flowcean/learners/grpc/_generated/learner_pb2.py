# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: learner.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rlearner.proto\"B\n\x0b\x44\x61taPackage\x12\x18\n\x06inputs\x18\x01 \x03(\x0b\x32\x08.DataRow\x12\x19\n\x07outputs\x18\x02 \x03(\x0b\x32\x08.DataRow\"K\n\nPrediction\x12\x1d\n\x0bpredictions\x18\x01 \x03(\x0b\x32\x08.DataRow\x12\x1e\n\x06status\x18\x02 \x01(\x0b\x32\x0e.StatusMessage\"%\n\x07\x44\x61taRow\x12\x1a\n\x06\x66ields\x18\x01 \x03(\x0b\x32\n.DataField\"\xc9\x01\n\tDataField\x12\r\n\x03int\x18\x01 \x01(\x05H\x00\x12\x10\n\x06\x64ouble\x18\x02 \x01(\x01H\x00\x12 \n\nvector_int\x18\x03 \x01(\x0b\x32\n.VectorIntH\x00\x12&\n\rvector_double\x18\x04 \x01(\x0b\x32\r.VectorDoubleH\x00\x12 \n\nmatrix_int\x18\x05 \x01(\x0b\x32\n.MatrixIntH\x00\x12&\n\rmatrix_double\x18\x06 \x01(\x0b\x32\r.MatrixDoubleH\x00\x42\x07\n\x05\x66ield\"\x19\n\tVectorInt\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x05\"\x1c\n\x0cVectorDouble\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x01\"B\n\tMatrixInt\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x05\x12\x11\n\trow_count\x18\x02 \x01(\x05\x12\x14\n\x0c\x63olumn_count\x18\x03 \x01(\x05\"E\n\x0cMatrixDouble\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x01\x12\x11\n\trow_count\x18\x02 \x01(\x05\x12\x14\n\x0c\x63olumn_count\x18\x03 \x01(\x05\"h\n\rStatusMessage\x12\x17\n\x06status\x18\x01 \x01(\x0e\x32\x07.Status\x12\x1a\n\x08messages\x18\x02 \x03(\x0b\x32\x08.Message\x12\x15\n\x08progress\x18\x03 \x01(\x05H\x00\x88\x01\x01\x42\x0b\n\t_progress\"H\n\x07Message\x12\x1c\n\tlog_level\x18\x01 \x01(\x0e\x32\t.LogLevel\x12\x0e\n\x06sender\x18\x02 \x01(\t\x12\x0f\n\x07message\x18\x03 \x01(\t\"\x07\n\x05\x45mpty*Z\n\x06Status\x12\x14\n\x10STATUS_UNDEFINED\x10\x00\x12\x12\n\x0eSTATUS_RUNNING\x10\x01\x12\x13\n\x0fSTATUS_FINISHED\x10\x02\x12\x11\n\rSTATUS_FAILED\x10\x03*\x87\x01\n\x08LogLevel\x12\x16\n\x12LOGLEVEL_UNDEFINED\x10\x00\x12\x12\n\x0eLOGLEVEL_DEBUG\x10\x01\x12\x11\n\rLOGLEVEL_INFO\x10\x02\x12\x14\n\x10LOGLEVEL_WARNING\x10\x03\x12\x12\n\x0eLOGLEVEL_ERROR\x10\x04\x12\x12\n\x0eLOGLEVEL_FATAL\x10\x05\x32x\n\x07Learner\x12)\n\x05Train\x12\x0c.DataPackage\x1a\x0e.StatusMessage\"\x00\x30\x01\x12&\n\x07Predict\x12\x0c.DataPackage\x1a\x0b.Prediction\"\x00\x12\x1a\n\x06\x45xport\x12\x06.Empty\x1a\x06.Empty\"\x00\x42\x1a\n\x18io.flowcean.learner.grpcb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'learner_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\030io.flowcean.learner.grpc'
  _globals['_STATUS']._serialized_start=790
  _globals['_STATUS']._serialized_end=880
  _globals['_LOGLEVEL']._serialized_start=883
  _globals['_LOGLEVEL']._serialized_end=1018
  _globals['_DATAPACKAGE']._serialized_start=17
  _globals['_DATAPACKAGE']._serialized_end=83
  _globals['_PREDICTION']._serialized_start=85
  _globals['_PREDICTION']._serialized_end=160
  _globals['_DATAROW']._serialized_start=162
  _globals['_DATAROW']._serialized_end=199
  _globals['_DATAFIELD']._serialized_start=202
  _globals['_DATAFIELD']._serialized_end=403
  _globals['_VECTORINT']._serialized_start=405
  _globals['_VECTORINT']._serialized_end=430
  _globals['_VECTORDOUBLE']._serialized_start=432
  _globals['_VECTORDOUBLE']._serialized_end=460
  _globals['_MATRIXINT']._serialized_start=462
  _globals['_MATRIXINT']._serialized_end=528
  _globals['_MATRIXDOUBLE']._serialized_start=530
  _globals['_MATRIXDOUBLE']._serialized_end=599
  _globals['_STATUSMESSAGE']._serialized_start=601
  _globals['_STATUSMESSAGE']._serialized_end=705
  _globals['_MESSAGE']._serialized_start=707
  _globals['_MESSAGE']._serialized_end=779
  _globals['_EMPTY']._serialized_start=781
  _globals['_EMPTY']._serialized_end=788
  _globals['_LEARNER']._serialized_start=1020
  _globals['_LEARNER']._serialized_end=1140
# @@protoc_insertion_point(module_scope)
