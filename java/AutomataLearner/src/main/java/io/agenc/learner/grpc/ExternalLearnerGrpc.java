package io.agenc.learner.grpc;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
//@javax.annotation.Generated(
//    value = "by gRPC proto compiler (version 1.56.0)",
//    comments = "Source: proto/grpcLearner.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class ExternalLearnerGrpc {

  private ExternalLearnerGrpc() {}

  public static final String SERVICE_NAME = "ExternalLearner";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.DataPackage,
      io.agenc.learner.grpc.GrpcLearner.StatusMessage> getTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Train",
      requestType = io.agenc.learner.grpc.GrpcLearner.DataPackage.class,
      responseType = io.agenc.learner.grpc.GrpcLearner.StatusMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
  public static io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.DataPackage,
      io.agenc.learner.grpc.GrpcLearner.StatusMessage> getTrainMethod() {
    io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.DataPackage, io.agenc.learner.grpc.GrpcLearner.StatusMessage> getTrainMethod;
    if ((getTrainMethod = ExternalLearnerGrpc.getTrainMethod) == null) {
      synchronized (ExternalLearnerGrpc.class) {
        if ((getTrainMethod = ExternalLearnerGrpc.getTrainMethod) == null) {
          ExternalLearnerGrpc.getTrainMethod = getTrainMethod =
              io.grpc.MethodDescriptor.<io.agenc.learner.grpc.GrpcLearner.DataPackage, io.agenc.learner.grpc.GrpcLearner.StatusMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Train"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.learner.grpc.GrpcLearner.DataPackage.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.learner.grpc.GrpcLearner.StatusMessage.getDefaultInstance()))
              .setSchemaDescriptor(new ExternalLearnerMethodDescriptorSupplier("Train"))
              .build();
        }
      }
    }
    return getTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.DataPackage,
      io.agenc.learner.grpc.GrpcLearner.Prediction> getPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Predict",
      requestType = io.agenc.learner.grpc.GrpcLearner.DataPackage.class,
      responseType = io.agenc.learner.grpc.GrpcLearner.Prediction.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.DataPackage,
      io.agenc.learner.grpc.GrpcLearner.Prediction> getPredictMethod() {
    io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.DataPackage, io.agenc.learner.grpc.GrpcLearner.Prediction> getPredictMethod;
    if ((getPredictMethod = ExternalLearnerGrpc.getPredictMethod) == null) {
      synchronized (ExternalLearnerGrpc.class) {
        if ((getPredictMethod = ExternalLearnerGrpc.getPredictMethod) == null) {
          ExternalLearnerGrpc.getPredictMethod = getPredictMethod =
              io.grpc.MethodDescriptor.<io.agenc.learner.grpc.GrpcLearner.DataPackage, io.agenc.learner.grpc.GrpcLearner.Prediction>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Predict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.learner.grpc.GrpcLearner.DataPackage.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.learner.grpc.GrpcLearner.Prediction.getDefaultInstance()))
              .setSchemaDescriptor(new ExternalLearnerMethodDescriptorSupplier("Predict"))
              .build();
        }
      }
    }
    return getPredictMethod;
  }

  private static volatile io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.Empty,
      io.agenc.learner.grpc.GrpcLearner.Empty> getExportMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Export",
      requestType = io.agenc.learner.grpc.GrpcLearner.Empty.class,
      responseType = io.agenc.learner.grpc.GrpcLearner.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.Empty,
      io.agenc.learner.grpc.GrpcLearner.Empty> getExportMethod() {
    io.grpc.MethodDescriptor<io.agenc.learner.grpc.GrpcLearner.Empty, io.agenc.learner.grpc.GrpcLearner.Empty> getExportMethod;
    if ((getExportMethod = ExternalLearnerGrpc.getExportMethod) == null) {
      synchronized (ExternalLearnerGrpc.class) {
        if ((getExportMethod = ExternalLearnerGrpc.getExportMethod) == null) {
          ExternalLearnerGrpc.getExportMethod = getExportMethod =
              io.grpc.MethodDescriptor.<io.agenc.learner.grpc.GrpcLearner.Empty, io.agenc.learner.grpc.GrpcLearner.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Export"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.learner.grpc.GrpcLearner.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.learner.grpc.GrpcLearner.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new ExternalLearnerMethodDescriptorSupplier("Export"))
              .build();
        }
      }
    }
    return getExportMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static ExternalLearnerStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ExternalLearnerStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ExternalLearnerStub>() {
        @java.lang.Override
        public ExternalLearnerStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ExternalLearnerStub(channel, callOptions);
        }
      };
    return ExternalLearnerStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static ExternalLearnerBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ExternalLearnerBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ExternalLearnerBlockingStub>() {
        @java.lang.Override
        public ExternalLearnerBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ExternalLearnerBlockingStub(channel, callOptions);
        }
      };
    return ExternalLearnerBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static ExternalLearnerFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ExternalLearnerFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ExternalLearnerFutureStub>() {
        @java.lang.Override
        public ExternalLearnerFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ExternalLearnerFutureStub(channel, callOptions);
        }
      };
    return ExternalLearnerFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void train(io.agenc.learner.grpc.GrpcLearner.DataPackage request,
        io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.StatusMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getTrainMethod(), responseObserver);
    }

    /**
     */
    default void predict(io.agenc.learner.grpc.GrpcLearner.DataPackage request,
        io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Prediction> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictMethod(), responseObserver);
    }

    /**
     */
    default void export(io.agenc.learner.grpc.GrpcLearner.Empty request,
        io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getExportMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service ExternalLearner.
   */
  public static abstract class ExternalLearnerImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return ExternalLearnerGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service ExternalLearner.
   */
  public static final class ExternalLearnerStub
      extends io.grpc.stub.AbstractAsyncStub<ExternalLearnerStub> {
    private ExternalLearnerStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ExternalLearnerStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ExternalLearnerStub(channel, callOptions);
    }

    /**
     */
    public void train(io.agenc.learner.grpc.GrpcLearner.DataPackage request,
        io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.StatusMessage> responseObserver) {
      io.grpc.stub.ClientCalls.asyncServerStreamingCall(
          getChannel().newCall(getTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predict(io.agenc.learner.grpc.GrpcLearner.DataPackage request,
        io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Prediction> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void export(io.agenc.learner.grpc.GrpcLearner.Empty request,
        io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getExportMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service ExternalLearner.
   */
  public static final class ExternalLearnerBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<ExternalLearnerBlockingStub> {
    private ExternalLearnerBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ExternalLearnerBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ExternalLearnerBlockingStub(channel, callOptions);
    }

    /**
     */
    public java.util.Iterator<io.agenc.learner.grpc.GrpcLearner.StatusMessage> train(
        io.agenc.learner.grpc.GrpcLearner.DataPackage request) {
      return io.grpc.stub.ClientCalls.blockingServerStreamingCall(
          getChannel(), getTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public io.agenc.learner.grpc.GrpcLearner.Prediction predict(io.agenc.learner.grpc.GrpcLearner.DataPackage request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictMethod(), getCallOptions(), request);
    }

    /**
     */
    public io.agenc.learner.grpc.GrpcLearner.Empty export(io.agenc.learner.grpc.GrpcLearner.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getExportMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service ExternalLearner.
   */
  public static final class ExternalLearnerFutureStub
      extends io.grpc.stub.AbstractFutureStub<ExternalLearnerFutureStub> {
    private ExternalLearnerFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ExternalLearnerFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ExternalLearnerFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<io.agenc.learner.grpc.GrpcLearner.Prediction> predict(
        io.agenc.learner.grpc.GrpcLearner.DataPackage request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<io.agenc.learner.grpc.GrpcLearner.Empty> export(
        io.agenc.learner.grpc.GrpcLearner.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getExportMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_TRAIN = 0;
  private static final int METHODID_PREDICT = 1;
  private static final int METHODID_EXPORT = 2;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final AsyncService serviceImpl;
    private final int methodId;

    MethodHandlers(AsyncService serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_TRAIN:
          serviceImpl.train((io.agenc.learner.grpc.GrpcLearner.DataPackage) request,
              (io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.StatusMessage>) responseObserver);
          break;
        case METHODID_PREDICT:
          serviceImpl.predict((io.agenc.learner.grpc.GrpcLearner.DataPackage) request,
              (io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Prediction>) responseObserver);
          break;
        case METHODID_EXPORT:
          serviceImpl.export((io.agenc.learner.grpc.GrpcLearner.Empty) request,
              (io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Empty>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getTrainMethod(),
          io.grpc.stub.ServerCalls.asyncServerStreamingCall(
            new MethodHandlers<
              io.agenc.learner.grpc.GrpcLearner.DataPackage,
              io.agenc.learner.grpc.GrpcLearner.StatusMessage>(
                service, METHODID_TRAIN)))
        .addMethod(
          getPredictMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              io.agenc.learner.grpc.GrpcLearner.DataPackage,
              io.agenc.learner.grpc.GrpcLearner.Prediction>(
                service, METHODID_PREDICT)))
        .addMethod(
          getExportMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              io.agenc.learner.grpc.GrpcLearner.Empty,
              io.agenc.learner.grpc.GrpcLearner.Empty>(
                service, METHODID_EXPORT)))
        .build();
  }

  private static abstract class ExternalLearnerBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    ExternalLearnerBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return io.agenc.learner.grpc.GrpcLearner.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("ExternalLearner");
    }
  }

  private static final class ExternalLearnerFileDescriptorSupplier
      extends ExternalLearnerBaseDescriptorSupplier {
    ExternalLearnerFileDescriptorSupplier() {}
  }

  private static final class ExternalLearnerMethodDescriptorSupplier
      extends ExternalLearnerBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    ExternalLearnerMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (ExternalLearnerGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new ExternalLearnerFileDescriptorSupplier())
              .addMethod(getTrainMethod())
              .addMethod(getPredictMethod())
              .addMethod(getExportMethod())
              .build();
        }
      }
    }
    return result;
  }
}
