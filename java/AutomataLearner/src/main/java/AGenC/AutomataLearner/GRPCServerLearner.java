package AGenC.AutomataLearner;

import io.agenc.learner.grpc.LearnerOuterClass.LogLevel;
import io.agenc.learner.grpc.LearnerOuterClass.Message;
import io.agenc.learner.grpc.LearnerOuterClass.DataRow;
import io.agenc.learner.grpc.LearnerOuterClass.DataField;
import io.agenc.learner.grpc.LearnerOuterClass.Prediction;
import io.agenc.learner.grpc.LearnerOuterClass.Prediction.Builder;
import io.agenc.learner.grpc.LearnerOuterClass.Status;
import io.agenc.learner.grpc.LearnerOuterClass.StatusMessage;
import io.agenc.learner.grpc.LearnerOuterClass.VectorInt;
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
import io.agenc.learner.grpc.LearnerGrpc;

public class GRPCServerLearner extends LearnerGrpc.LearnerImplBase {
	private MealyMachine<?, Integer, ?, Integer> model;

	// TODO: add Docs here and in tool docu
	public void train(io.agenc.learner.grpc.LearnerOuterClass.DataPackage data,
			io.grpc.stub.StreamObserver<io.agenc.learner.grpc.LearnerOuterClass.StatusMessage> responseObserver) {

		StatusMessage message = buildMessage(Status.STATUS_RUNNING, "Started Processing",LogLevel.LOGLEVEL_INFO);
		responseObserver.onNext(message);

		List<DataRow> inputs = data.getInputsList();
		List<DataRow> outputs = data.getOutputsList();

		responseObserver.onNext(message);

		List<Pair<Word<Integer>, Word<Integer>>> wordObs;
		try {
			wordObs = exportDataRowAsWords(inputs, outputs);
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

	public void predict(io.agenc.learner.grpc.LearnerOuterClass.DataPackage data,
			io.grpc.stub.StreamObserver<io.agenc.learner.grpc.LearnerOuterClass.Prediction> responseObserver) {
		
		Builder pred = Prediction.newBuilder();

		List<Word<Integer>> wordObs;
		try {
			wordObs = exportAsWords(data.getInputsList());
		} catch (MessageException e) {
			io.agenc.learner.grpc.LearnerOuterClass.DataRow emptyObs = DataRow.newBuilder().build();
			pred.setStatus(buildMessage(Status.STATUS_FAILED, e.getMessage(),LogLevel.LOGLEVEL_FATAL));
			pred.addPredictions(emptyObs);
			responseObserver.onNext(pred.build());
			responseObserver.onCompleted();
			return;
		}

		for (Word<Integer> input : wordObs) {
			Word<Integer> pred_output = model.computeOutput(Word.fromWords(input));
			List<DataField> fields = new ArrayList<DataField>();
			for(int val : pred_output) {
				fields.add(DataField.newBuilder().setInt(val).build());
			}
			io.agenc.learner.grpc.LearnerOuterClass.DataRow.Builder obs = DataRow.newBuilder().addAllFields(fields);
			pred.addPredictions(obs.build());
		}

		responseObserver.onNext(pred.build());
		responseObserver.onCompleted();
	}

	public void export(io.agenc.learner.grpc.LearnerOuterClass.Empty request,
			io.grpc.stub.StreamObserver<io.agenc.learner.grpc.LearnerOuterClass.Empty> responseObserver) {
		//TODO
	}
	
	List<Word<Integer>> exportAsWords(List<DataRow> datarows) throws MessageException{
		
		List<Word<Integer>> words = new ArrayList<Word<Integer>>();
		
		for (DataRow row : datarows) {
			List<DataField> fields = row.getFieldsList();
			
			//TODO: add other types of inputs
			Word<Integer> word = Word.epsilon();
			for(DataField field : fields) {
				if (field.hasInt()) { //TODO: add other types of inputs
					word = word.append(field.getInt());
				} else {
					throw new MessageException("Input and Output have to be of type Int");
				}
			}
			words.add(word);
		}
		return words;
	}

	List<Pair<Word<Integer>, Word<Integer>>> exportDataRowAsWords(List<DataRow> inputs, List<DataRow> outputs) throws MessageException {
		List<Pair<Word<Integer>, Word<Integer>>> wordObs = new ArrayList<Pair<Word<Integer>, Word<Integer>>>();
		//number of inputs / outputs equals number of observations / sequences
		assert(inputs.size() == outputs.size()); //TODO: change assert to throw of MessageException?
		
		List<Word<Integer>> inputWords = exportAsWords(inputs);
		List<Word<Integer>> outputWords = exportAsWords(outputs);

		for (int i = 0; i < inputWords.size(); i++) {
			Word<Integer> input = inputWords.get(i), output = outputWords.get(i);
			assert(input.size() == output.size()); // number of samples within an observation / sequence
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
