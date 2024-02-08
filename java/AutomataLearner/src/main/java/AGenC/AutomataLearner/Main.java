package AGenC.AutomataLearner;

import io.grpc.Server;
import io.grpc.ServerBuilder;

import java.io.IOException;

public final class Main {

    private Main() {
        // prevent instantiation
    }

    /**
     * Initial starting point. Instantiation of GRPC-Server.
     *
     * @param args Command Line Arguments
     * @throws IOException termination await exception
     */
    public static void main(String[] args) throws IOException {
        Server server = ServerBuilder.forPort(8080).maxInboundMessageSize(Integer.MAX_VALUE)
                .addService(new GRPCServerLearner()).build();

        server.start();
        try {
            server.awaitTermination();
        } catch (InterruptedException e) {
            System.out.println("Awaiting server termination failed: " + e.getMessage());
        }

    }
}
