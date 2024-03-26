package flowcean.AutomataLearner;

import io.flowcean.learner.grpc.LearnerOuterClass.LogLevel;
import io.flowcean.learner.grpc.LearnerOuterClass.Message;
import io.flowcean.learner.grpc.LearnerOuterClass.Status;
import io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage;

/**
 * Exception class for Status Messages
 */
public class MessageException extends Exception {
    private static final long serialVersionUID = 1L;
    private final StatusMessage message;

    /**
     * Message Exception Constructor
     *
     * @param message the message of the status
     */
    public MessageException(String message) {
        super(message);
        this.message = StatusMessage.newBuilder()
                .setStatus(Status.STATUS_FAILED)
                .addMessages(Message
                        .newBuilder()
                        .setMessage(message)
                        .setLogLevel(LogLevel.LOGLEVEL_FATAL))
                .build();
    }

    public StatusMessage getStatusMessage() {
        return message;
    }

}
