package AGenC.AutomataLearner;

import io.agenc.learner.grpc.GrpcLearner.ColumnMetadata;
import io.agenc.learner.grpc.GrpcLearner.FeatureType;
import io.agenc.learner.grpc.GrpcLearner.LogLevel;
import io.agenc.learner.grpc.GrpcLearner.Message;
import io.agenc.learner.grpc.GrpcLearner.Observation;
import io.agenc.learner.grpc.GrpcLearner.ObservationField;
import io.agenc.learner.grpc.GrpcLearner.Prediction;
import io.agenc.learner.grpc.GrpcLearner.Prediction.Builder;
import io.agenc.learner.grpc.GrpcLearner.Status;
import io.agenc.learner.grpc.GrpcLearner.StatusMessage;
import io.agenc.learner.grpc.GrpcLearner.VectorInt;
import net.automatalib.automata.transducers.MealyMachine;
import net.automatalib.words.Alphabet;
import net.automatalib.words.Word;
import net.automatalib.words.impl.Alphabets;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.javatuples.Pair;

import de.learnlib.algorithms.rpni.BlueFringeRPNIMealy;
import de.learnlib.api.algorithm.PassiveLearningAlgorithm.PassiveMealyLearner;
import io.agenc.learner.grpc.ExternalLearnerGrpc;

public class GRPCServerLearner extends ExternalLearnerGrpc.ExternalLearnerImplBase {
	private MealyMachine<?, Integer, ?, Integer> model;

	// TODO: add Docs here and in tool docu
	public void train(io.agenc.learner.grpc.GrpcLearner.DataPackage data,
			io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.StatusMessage> responseObserver) {

		StatusMessage message = buildMessage(Status.STATUS_RUNNING, "Started Processing",LogLevel.LOGLEVEL_INFO);
		responseObserver.onNext(message);

		List<ColumnMetadata> metadata = data.getMetadataList();
		List<Observation> observations = data.getObservationsList();

		// Find input and output index
		Pair<Integer, Integer> indices;
		try {
			indices = findInputOutputIndex(metadata);
		} catch (MessageException e) {
			responseObserver.onNext(e.getStatusMessage());
			responseObserver.onCompleted();
			return;
		}

		responseObserver.onNext(message);
		responseObserver.onCompleted();

		List<Pair<Word<Integer>, Word<Integer>>> wordObs;
		try {
			wordObs = exportObservationsAsWords(observations, indices);
		} catch (MessageException e) {
			responseObserver.onNext(e.getStatusMessage());
			responseObserver.onCompleted();
			return;
		}
		Alphabet<Integer> inputAlphabet = extractAlphabet(wordObs);

		final PassiveMealyLearner<Integer, Integer> learner = new BlueFringeRPNIMealy<>(inputAlphabet);
		for (Pair<Word<Integer>, Word<Integer>> pair : wordObs) {
			message = buildMessage(Status.STATUS_RUNNING, "Inferring Observations",LogLevel.LOGLEVEL_INFO);
			responseObserver.onNext(message);
			learner.addSample(pair.getValue0(), pair.getValue1());
		}

		model = learner.computeModel();

		message = buildMessage(Status.STATUS_FINISHED, "Model built successfully",LogLevel.LOGLEVEL_INFO);
		responseObserver.onNext(message);
		responseObserver.onCompleted();
	}

	public void predict(io.agenc.learner.grpc.GrpcLearner.DataPackage data,
			io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Prediction> responseObserver) {
		
		Builder pred = Prediction.newBuilder();
		Pair<Integer, Integer> indices;
		try {
			indices = findInputOutputIndex(data.getMetadataList());
		} catch (MessageException e1) {
			io.agenc.learner.grpc.GrpcLearner.Observation emptyObs = Observation.newBuilder().build();
			pred.addPredictions(emptyObs);
			pred.setStatus(buildMessage(Status.STATUS_FAILED, e1.getMessage(),LogLevel.LOGLEVEL_FATAL));
			responseObserver.onNext(pred.build());
			responseObserver.onCompleted();
			return;
		}

		List<Pair<Word<Integer>, Word<Integer>>> wordObs;
		try {
			wordObs = exportObservationsAsWords(data.getObservationsList(), indices);
		} catch (MessageException e) {
			io.agenc.learner.grpc.GrpcLearner.Observation emptyObs = Observation.newBuilder().build();
			pred.setStatus(buildMessage(Status.STATUS_FAILED, e.getMessage(),LogLevel.LOGLEVEL_FATAL));
			pred.addPredictions(emptyObs);
			responseObserver.onNext(pred.build());
			responseObserver.onCompleted();
			return;
		}

		int correct = 0;
		for (Pair<Word<Integer>, Word<Integer>> pair : wordObs) {
			Word<Integer> input = pair.getValue0();
			Word<Integer> output = pair.getValue1();
			Word<Integer> pred_output = model.computeOutput(Word.fromWords(input));
			io.agenc.learner.grpc.GrpcLearner.Observation.Builder obs = Observation.newBuilder().addFields(
					ObservationField.newBuilder().setVectorInt(VectorInt.newBuilder().addAllData(pred_output)));
			pred.addPredictions(obs.build());
			if (pred_output.equals(output)) {
				correct++;
			}
		}

		System.out.println("Correct predictions: " + correct);

		responseObserver.onNext(pred.build());
		responseObserver.onCompleted();
	}

	public void export(io.agenc.learner.grpc.GrpcLearner.Empty request,
			io.grpc.stub.StreamObserver<io.agenc.learner.grpc.GrpcLearner.Empty> responseObserver) {
		//TODO
	}

	Pair<Integer, Integer> findInputOutputIndex(List<ColumnMetadata> metadata) throws MessageException {
		int input_index = -1, output_index = -1;
		for (int i = 0; i < metadata.size(); i++) {
			if (metadata.get(i).getFeatureType() == FeatureType.FEATURETYPE_INPUT) {
				if (input_index != -1) {
					throw new MessageException("Multiple Definition of Input");
				} else {
					input_index = i;
				}
			} else if (metadata.get(i).getFeatureType() == FeatureType.FEATURETYPE_TARGET) {
				if (output_index != -1) {
					throw new MessageException("Multiple Definition of Output");
				} else {
					output_index = i;
				}
			}
		}

		if (output_index == -1 || input_index == -1) {
			throw new MessageException("Missing Definition of in- or output");
		}
		return new Pair<>(input_index, output_index);
	}

	List<Pair<Word<Integer>, Word<Integer>>> exportObservationsAsWords(List<Observation> observations,
			Pair<Integer, Integer> indices) throws MessageException {
		List<Pair<Word<Integer>, Word<Integer>>> wordObs = new ArrayList<Pair<Word<Integer>, Word<Integer>>>();
		int input_index = indices.getValue0();
		int output_index = indices.getValue1();

		for (Observation obs : observations) {
			List<ObservationField> fields = obs.getFieldsList();
			ObservationField inputField = fields.get(input_index);
			ObservationField outputField = fields.get(output_index);

			Word<Integer> input, output;
			if (inputField.hasInt() && outputField.hasInt()) {
				input = Word.fromLetter(inputField.getInt());
				output = Word.fromLetter(outputField.getInt());
			} else if (inputField.hasVectorInt() && outputField.hasVectorInt()) {
				input = Word.fromList(inputField.getVectorInt().getDataList());
				output = Word.fromList(inputField.getVectorInt().getDataList());
			} else {
				throw new MessageException("Input and Output have to be of same type from Int or VectorInt");
			}
			wordObs.add(new Pair<>(input, output));
		}
		return wordObs;
	}

	Alphabet<Integer> extractAlphabet(List<Pair<Word<Integer>, Word<Integer>>> wordObs) {
		Set<Integer> inputSet = new HashSet<Integer>();

		for (Pair<Word<Integer>, Word<Integer>> observation : wordObs) {
			for (int symbol : observation.getValue0()) {
				inputSet.add(symbol);
			}
		}

		return Alphabets.fromCollection(inputSet);
	}

	StatusMessage buildMessage(Status status, String message, LogLevel level) {
		return StatusMessage.newBuilder().setStatus(status).addMessages(Message.newBuilder().setMessage(message).setLogLevel(level))
				.build();
	}

}
