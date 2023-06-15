package io.agenc.externalframework;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.56.0)",
    comments = "Source: externalFramework.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class ExternalLearnerGrpc {

  private ExternalLearnerGrpc() {}

  public static final String SERVICE_NAME = "ExternalLearner";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<io.agenc.externalframework.ExternalFramework.DataPackage,
      io.agenc.externalframework.ExternalFramework.LearnerData> getTrainLearnerMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "TrainLearner",
      requestType = io.agenc.externalframework.ExternalFramework.DataPackage.class,
      responseType = io.agenc.externalframework.ExternalFramework.LearnerData.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<io.agenc.externalframework.ExternalFramework.DataPackage,
      io.agenc.externalframework.ExternalFramework.LearnerData> getTrainLearnerMethod() {
    io.grpc.MethodDescriptor<io.agenc.externalframework.ExternalFramework.DataPackage, io.agenc.externalframework.ExternalFramework.LearnerData> getTrainLearnerMethod;
    if ((getTrainLearnerMethod = ExternalLearnerGrpc.getTrainLearnerMethod) == null) {
      synchronized (ExternalLearnerGrpc.class) {
        if ((getTrainLearnerMethod = ExternalLearnerGrpc.getTrainLearnerMethod) == null) {
          ExternalLearnerGrpc.getTrainLearnerMethod = getTrainLearnerMethod =
              io.grpc.MethodDescriptor.<io.agenc.externalframework.ExternalFramework.DataPackage, io.agenc.externalframework.ExternalFramework.LearnerData>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "TrainLearner"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.externalframework.ExternalFramework.DataPackage.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.externalframework.ExternalFramework.LearnerData.getDefaultInstance()))
              .setSchemaDescriptor(new ExternalLearnerMethodDescriptorSupplier("TrainLearner"))
              .build();
        }
      }
    }
    return getTrainLearnerMethod;
  }

  private static volatile io.grpc.MethodDescriptor<io.agenc.externalframework.ExternalFramework.DataRow,
      io.agenc.externalframework.ExternalFramework.DataRow> getRunLearnerMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "RunLearner",
      requestType = io.agenc.externalframework.ExternalFramework.DataRow.class,
      responseType = io.agenc.externalframework.ExternalFramework.DataRow.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<io.agenc.externalframework.ExternalFramework.DataRow,
      io.agenc.externalframework.ExternalFramework.DataRow> getRunLearnerMethod() {
    io.grpc.MethodDescriptor<io.agenc.externalframework.ExternalFramework.DataRow, io.agenc.externalframework.ExternalFramework.DataRow> getRunLearnerMethod;
    if ((getRunLearnerMethod = ExternalLearnerGrpc.getRunLearnerMethod) == null) {
      synchronized (ExternalLearnerGrpc.class) {
        if ((getRunLearnerMethod = ExternalLearnerGrpc.getRunLearnerMethod) == null) {
          ExternalLearnerGrpc.getRunLearnerMethod = getRunLearnerMethod =
              io.grpc.MethodDescriptor.<io.agenc.externalframework.ExternalFramework.DataRow, io.agenc.externalframework.ExternalFramework.DataRow>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "RunLearner"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.externalframework.ExternalFramework.DataRow.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.agenc.externalframework.ExternalFramework.DataRow.getDefaultInstance()))
              .setSchemaDescriptor(new ExternalLearnerMethodDescriptorSupplier("RunLearner"))
              .build();
        }
      }
    }
    return getRunLearnerMethod;
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
    default void trainLearner(io.agenc.externalframework.ExternalFramework.DataPackage request,
        io.grpc.stub.StreamObserver<io.agenc.externalframework.ExternalFramework.LearnerData> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getTrainLearnerMethod(), responseObserver);
    }

    /**
     */
    default void runLearner(io.agenc.externalframework.ExternalFramework.DataRow request,
        io.grpc.stub.StreamObserver<io.agenc.externalframework.ExternalFramework.DataRow> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRunLearnerMethod(), responseObserver);
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
    public void trainLearner(io.agenc.externalframework.ExternalFramework.DataPackage request,
        io.grpc.stub.StreamObserver<io.agenc.externalframework.ExternalFramework.LearnerData> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getTrainLearnerMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void runLearner(io.agenc.externalframework.ExternalFramework.DataRow request,
        io.grpc.stub.StreamObserver<io.agenc.externalframework.ExternalFramework.DataRow> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRunLearnerMethod(), getCallOptions()), request, responseObserver);
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
    public io.agenc.externalframework.ExternalFramework.LearnerData trainLearner(io.agenc.externalframework.ExternalFramework.DataPackage request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getTrainLearnerMethod(), getCallOptions(), request);
    }

    /**
     */
    public io.agenc.externalframework.ExternalFramework.DataRow runLearner(io.agenc.externalframework.ExternalFramework.DataRow request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRunLearnerMethod(), getCallOptions(), request);
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
    public com.google.common.util.concurrent.ListenableFuture<io.agenc.externalframework.ExternalFramework.LearnerData> trainLearner(
        io.agenc.externalframework.ExternalFramework.DataPackage request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getTrainLearnerMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<io.agenc.externalframework.ExternalFramework.DataRow> runLearner(
        io.agenc.externalframework.ExternalFramework.DataRow request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRunLearnerMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_TRAIN_LEARNER = 0;
  private static final int METHODID_RUN_LEARNER = 1;

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
        case METHODID_TRAIN_LEARNER:
          serviceImpl.trainLearner((io.agenc.externalframework.ExternalFramework.DataPackage) request,
              (io.grpc.stub.StreamObserver<io.agenc.externalframework.ExternalFramework.LearnerData>) responseObserver);
          break;
        case METHODID_RUN_LEARNER:
          serviceImpl.runLearner((io.agenc.externalframework.ExternalFramework.DataRow) request,
              (io.grpc.stub.StreamObserver<io.agenc.externalframework.ExternalFramework.DataRow>) responseObserver);
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
          getTrainLearnerMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              io.agenc.externalframework.ExternalFramework.DataPackage,
              io.agenc.externalframework.ExternalFramework.LearnerData>(
                service, METHODID_TRAIN_LEARNER)))
        .addMethod(
          getRunLearnerMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              io.agenc.externalframework.ExternalFramework.DataRow,
              io.agenc.externalframework.ExternalFramework.DataRow>(
                service, METHODID_RUN_LEARNER)))
        .build();
  }

  private static abstract class ExternalLearnerBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    ExternalLearnerBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return io.agenc.externalframework.ExternalFramework.getDescriptor();
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
              .addMethod(getTrainLearnerMethod())
              .addMethod(getRunLearnerMethod())
              .build();
        }
      }
    }
    return result;
  }
}
