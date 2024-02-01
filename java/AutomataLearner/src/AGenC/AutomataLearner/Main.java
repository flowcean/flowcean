package AGenC.AutomataLearner;

import java.io.IOException;

import io.grpc.Server;
import io.grpc.ServerBuilder;

public final class Main {

	private Main() {
		// prevent instantiation
	}

	public static void main(String[] args) throws IOException {
		// Instantiates server with larger message inbounds size
		Server server = ServerBuilder.forPort(8080).maxInboundMessageSize(Integer.MAX_VALUE)
				.addService(new GRPCServerLearner()).build();

		server.start();
		try {
			server.awaitTermination();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
