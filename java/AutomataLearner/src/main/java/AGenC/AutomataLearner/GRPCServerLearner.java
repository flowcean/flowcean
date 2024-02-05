package AGenC.AutomataLearner;

import io.agenc.learner.grpc.LearnerOuterClass.LogLevel;
import io.agenc.learner.grpc.LearnerOuterClass.Message;
import io.agenc.learner.grpc.LearnerOuterClass.DataRow;
import io.agenc.learner.grpc.LearnerOuterClass.DataField;
import io.agenc.learner.grpc.LearnerOuterClass.Prediction;
import io.agenc.learner.grpc.LearnerOuterClass.Prediction.Builder;
import io.agenc.learner.grpc.LearnerOuterClass.Status;
import io.agenc.learner.grpc.LearnerOuterClass.StatusMessage;
import net.automatalib.alphabet.Alphabet;
import net.automatalib.alphabet.Alphabets;
import net.automatalib.automaton.transducer.MealyMachine;
import net.automatalib.word.Word;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.javatuples.Pair;

import io.agenc.learner.grpc.LearnerGrpc;
import de.learnlib.algorithm.PassiveLearningAlgorithm.PassiveMealyLearner;
import de.learnlib.algorithm.rpni.BlueFringeRPNIMealy;

import javax.xml.crypto.Data;

public class GRPCServerLearner extends LearnerGrpc.LearnerImplBase {
	private MealyMachine<?, Integer, ?, Integer> model;

	// TODO: add Docs here and in tool docu
	public void train(io.agenc.learner.grpc.LearnerOuterClass.DataPackage data,
			io.grpc.stub.StreamObserver<io.agenc.learner.grpc.LearnerOuterClass.StatusMessage> responseObserver) {

		StatusMessage message = buildMessage(Status.STATUS_RUNNING, "Started Processing",LogLevel.LOGLEVEL_INFO);
		responseObserver.onNext(message);

		List<DataRow> rawwords = data.getInputsList();
		List<DataRow> rawoutputs = data.getOutputsList();

		responseObserver.onNext(message);

		List<Pair<Word<Integer>, Word<Integer>>> wordObs;
		try {
			wordObs = exportDataRowAsWords(rawwords, rawoutputs,true);
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

		List<DataRow> rawwords = data.getInputsList();
		List<DataRow> rawoutputs = data.getOutputsList();

		List<Pair<Word<Integer>, Word<Integer>>> wordObs;
		try {
			wordObs = exportDataRowAsWords(rawwords, rawoutputs,false);
		} catch (MessageException e) {
			io.agenc.learner.grpc.LearnerOuterClass.DataRow emptyObs = DataRow.newBuilder().build();
			pred.setStatus(buildMessage(Status.STATUS_FAILED, e.getMessage(),LogLevel.LOGLEVEL_FATAL));
			pred.addPredictions(emptyObs);
			responseObserver.onNext(pred.build());
			responseObserver.onCompleted();
			return;
		}


		for (Pair<Word<Integer>, Word<Integer>> pair : wordObs) {
			Word<Integer> pred_output = model.computeOutput(pair.getValue0());
			int val = pred_output.lastSymbol();
			List<DataField> fields = new ArrayList<DataField>();
			fields.add(DataField.newBuilder().setInt(val).build());
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

	List<Pair<Word<Integer>, Word<Integer>>> exportDataRowAsWords(List<DataRow> rawwords, List<DataRow> rawoutputs, Boolean withOut) throws MessageException {
		List<Pair<Word<Integer>, Word<Integer>>> wordObs = new ArrayList<Pair<Word<Integer>, Word<Integer>>>();
		//number of inputs / outputs equals number of observations / sequences

		for (int i = 0; i < rawwords.size(); i++) {
			DataRow row = rawwords.get(i);
			List<Integer> inputWord = new ArrayList<>();
			List<Integer> outputWord = new ArrayList<>();
			for(int j = 0; j < row.getFieldsCount(); j++){
				if(j % 2 == 0){
					inputWord.add(row.getFields(j).getInt());
				}
				else{
					outputWord.add(row.getFields(j).getInt());
				}
			}
			if(withOut)
				outputWord.add(rawoutputs.get(i).getFields(0).getInt());
			wordObs.add(new Pair<>(Word.fromList(inputWord), Word.fromList(outputWord)));
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
