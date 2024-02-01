package AGenC.AutomataLearner;

import io.agenc.learner.grpc.LearnerOuterClass.LogLevel;
import io.agenc.learner.grpc.LearnerOuterClass.Message;
import io.agenc.learner.grpc.LearnerOuterClass.Status;
import io.agenc.learner.grpc.LearnerOuterClass.StatusMessage;

public class MessageException extends Exception {
	private static final long serialVersionUID = 1L;
	private StatusMessage message;

    public MessageException(String message) {
        super(message);
        this.message = StatusMessage.newBuilder().setStatus(Status.STATUS_FAILED).addMessages(Message.newBuilder().setMessage(message).setLogLevel(LogLevel.LOGLEVEL_FATAL))
		.build();
    }

    public MessageException(String message, Throwable cause) {
        super(message, cause);
        this.message = StatusMessage.newBuilder().setStatus(Status.STATUS_FAILED).addMessages(Message.newBuilder().setMessage(message).setLogLevel(LogLevel.LOGLEVEL_FATAL))
        		.build();
    }

    public StatusMessage getStatusMessage() {
        return message;
    }
    
}
