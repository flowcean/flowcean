package io.flowcean.learner.grpc;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.66.0)",
    comments = "Source: learner.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class LearnerGrpc {

  private LearnerGrpc() {}

  public static final java.lang.String SERVICE_NAME = "Learner";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage,
      io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage> getTrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Train",
      requestType = io.flowcean.learner.grpc.LearnerOuterClass.DataPackage.class,
      responseType = io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage.class,
      methodType = io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
  public static io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage,
      io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage> getTrainMethod() {
    io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage, io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage> getTrainMethod;
    if ((getTrainMethod = LearnerGrpc.getTrainMethod) == null) {
      synchronized (LearnerGrpc.class) {
        if ((getTrainMethod = LearnerGrpc.getTrainMethod) == null) {
          LearnerGrpc.getTrainMethod = getTrainMethod =
              io.grpc.MethodDescriptor.<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage, io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Train"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.flowcean.learner.grpc.LearnerOuterClass.DataPackage.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage.getDefaultInstance()))
              .setSchemaDescriptor(new LearnerMethodDescriptorSupplier("Train"))
              .build();
        }
      }
    }
    return getTrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage,
      io.flowcean.learner.grpc.LearnerOuterClass.Prediction> getPredictMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Predict",
      requestType = io.flowcean.learner.grpc.LearnerOuterClass.DataPackage.class,
      responseType = io.flowcean.learner.grpc.LearnerOuterClass.Prediction.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage,
      io.flowcean.learner.grpc.LearnerOuterClass.Prediction> getPredictMethod() {
    io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage, io.flowcean.learner.grpc.LearnerOuterClass.Prediction> getPredictMethod;
    if ((getPredictMethod = LearnerGrpc.getPredictMethod) == null) {
      synchronized (LearnerGrpc.class) {
        if ((getPredictMethod = LearnerGrpc.getPredictMethod) == null) {
          LearnerGrpc.getPredictMethod = getPredictMethod =
              io.grpc.MethodDescriptor.<io.flowcean.learner.grpc.LearnerOuterClass.DataPackage, io.flowcean.learner.grpc.LearnerOuterClass.Prediction>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Predict"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.flowcean.learner.grpc.LearnerOuterClass.DataPackage.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.flowcean.learner.grpc.LearnerOuterClass.Prediction.getDefaultInstance()))
              .setSchemaDescriptor(new LearnerMethodDescriptorSupplier("Predict"))
              .build();
        }
      }
    }
    return getPredictMethod;
  }

  private static volatile io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.Empty,
      io.flowcean.learner.grpc.LearnerOuterClass.Empty> getExportMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Export",
      requestType = io.flowcean.learner.grpc.LearnerOuterClass.Empty.class,
      responseType = io.flowcean.learner.grpc.LearnerOuterClass.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.Empty,
      io.flowcean.learner.grpc.LearnerOuterClass.Empty> getExportMethod() {
    io.grpc.MethodDescriptor<io.flowcean.learner.grpc.LearnerOuterClass.Empty, io.flowcean.learner.grpc.LearnerOuterClass.Empty> getExportMethod;
    if ((getExportMethod = LearnerGrpc.getExportMethod) == null) {
      synchronized (LearnerGrpc.class) {
        if ((getExportMethod = LearnerGrpc.getExportMethod) == null) {
          LearnerGrpc.getExportMethod = getExportMethod =
              io.grpc.MethodDescriptor.<io.flowcean.learner.grpc.LearnerOuterClass.Empty, io.flowcean.learner.grpc.LearnerOuterClass.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Export"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.flowcean.learner.grpc.LearnerOuterClass.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  io.flowcean.learner.grpc.LearnerOuterClass.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new LearnerMethodDescriptorSupplier("Export"))
              .build();
        }
      }
    }
    return getExportMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static LearnerStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<LearnerStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<LearnerStub>() {
        @java.lang.Override
        public LearnerStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new LearnerStub(channel, callOptions);
        }
      };
    return LearnerStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static LearnerBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<LearnerBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<LearnerBlockingStub>() {
        @java.lang.Override
        public LearnerBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new LearnerBlockingStub(channel, callOptions);
        }
      };
    return LearnerBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static LearnerFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<LearnerFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<LearnerFutureStub>() {
        @java.lang.Override
        public LearnerFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new LearnerFutureStub(channel, callOptions);
        }
      };
    return LearnerFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void train(io.flowcean.learner.grpc.LearnerOuterClass.DataPackage request,
        io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getTrainMethod(), responseObserver);
    }

    /**
     */
    default void predict(io.flowcean.learner.grpc.LearnerOuterClass.DataPackage request,
        io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Prediction> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPredictMethod(), responseObserver);
    }

    /**
     */
    default void export(io.flowcean.learner.grpc.LearnerOuterClass.Empty request,
        io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getExportMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service Learner.
   */
  public static abstract class LearnerImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return LearnerGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service Learner.
   */
  public static final class LearnerStub
      extends io.grpc.stub.AbstractAsyncStub<LearnerStub> {
    private LearnerStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected LearnerStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new LearnerStub(channel, callOptions);
    }

    /**
     */
    public void train(io.flowcean.learner.grpc.LearnerOuterClass.DataPackage request,
        io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage> responseObserver) {
      io.grpc.stub.ClientCalls.asyncServerStreamingCall(
          getChannel().newCall(getTrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void predict(io.flowcean.learner.grpc.LearnerOuterClass.DataPackage request,
        io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Prediction> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void export(io.flowcean.learner.grpc.LearnerOuterClass.Empty request,
        io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getExportMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service Learner.
   */
  public static final class LearnerBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<LearnerBlockingStub> {
    private LearnerBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected LearnerBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new LearnerBlockingStub(channel, callOptions);
    }

    /**
     */
    public java.util.Iterator<io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage> train(
        io.flowcean.learner.grpc.LearnerOuterClass.DataPackage request) {
      return io.grpc.stub.ClientCalls.blockingServerStreamingCall(
          getChannel(), getTrainMethod(), getCallOptions(), request);
    }

    /**
     */
    public io.flowcean.learner.grpc.LearnerOuterClass.Prediction predict(io.flowcean.learner.grpc.LearnerOuterClass.DataPackage request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPredictMethod(), getCallOptions(), request);
    }

    /**
     */
    public io.flowcean.learner.grpc.LearnerOuterClass.Empty export(io.flowcean.learner.grpc.LearnerOuterClass.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getExportMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service Learner.
   */
  public static final class LearnerFutureStub
      extends io.grpc.stub.AbstractFutureStub<LearnerFutureStub> {
    private LearnerFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected LearnerFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new LearnerFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<io.flowcean.learner.grpc.LearnerOuterClass.Prediction> predict(
        io.flowcean.learner.grpc.LearnerOuterClass.DataPackage request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPredictMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<io.flowcean.learner.grpc.LearnerOuterClass.Empty> export(
        io.flowcean.learner.grpc.LearnerOuterClass.Empty request) {
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
          serviceImpl.train((io.flowcean.learner.grpc.LearnerOuterClass.DataPackage) request,
              (io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage>) responseObserver);
          break;
        case METHODID_PREDICT:
          serviceImpl.predict((io.flowcean.learner.grpc.LearnerOuterClass.DataPackage) request,
              (io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Prediction>) responseObserver);
          break;
        case METHODID_EXPORT:
          serviceImpl.export((io.flowcean.learner.grpc.LearnerOuterClass.Empty) request,
              (io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Empty>) responseObserver);
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
              io.flowcean.learner.grpc.LearnerOuterClass.DataPackage,
              io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage>(
                service, METHODID_TRAIN)))
        .addMethod(
          getPredictMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              io.flowcean.learner.grpc.LearnerOuterClass.DataPackage,
              io.flowcean.learner.grpc.LearnerOuterClass.Prediction>(
                service, METHODID_PREDICT)))
        .addMethod(
          getExportMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              io.flowcean.learner.grpc.LearnerOuterClass.Empty,
              io.flowcean.learner.grpc.LearnerOuterClass.Empty>(
                service, METHODID_EXPORT)))
        .build();
  }

  private static abstract class LearnerBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    LearnerBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return io.flowcean.learner.grpc.LearnerOuterClass.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("Learner");
    }
  }

  private static final class LearnerFileDescriptorSupplier
      extends LearnerBaseDescriptorSupplier {
    LearnerFileDescriptorSupplier() {}
  }

  private static final class LearnerMethodDescriptorSupplier
      extends LearnerBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    LearnerMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (LearnerGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new LearnerFileDescriptorSupplier())
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
