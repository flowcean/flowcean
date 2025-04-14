package flowcean.AutomataLearner;

import de.learnlib.algorithm.PassiveLearningAlgorithm.PassiveMealyLearner;
import de.learnlib.algorithm.rpni.BlueFringeRPNIMealy;
import io.flowcean.learner.grpc.LearnerGrpc;
import io.flowcean.learner.grpc.LearnerOuterClass;
import io.flowcean.learner.grpc.LearnerOuterClass.*;
import io.flowcean.learner.grpc.LearnerOuterClass.Prediction.Builder;
import net.automatalib.alphabet.Alphabet;
import net.automatalib.alphabet.Alphabets;
import net.automatalib.automaton.transducer.MealyMachine;
import net.automatalib.word.Word;
import org.javatuples.Pair;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Instantiation of the GRPC-Learner class defined by the protobuf protocol.
 */
public class GRPCServerLearner extends LearnerGrpc.LearnerImplBase {
    private MealyMachine<?, Integer, ?, Integer> model;

    /**
     * Train a MealyMachine model
     * 1. Read data
     * 2. Transform Data to Words
     * 3. Extract input Alphabet
     * 4. Learn Model
     *
     * @param data             GRPC DataPackage, rows are traces where entries are alternating inputs and outputs
     * @param responseObserver Interface to transmit status messages via the GRPC connection
     */
    public void train(io.flowcean.learner.grpc.LearnerOuterClass.DataPackage data,
                      io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.StatusMessage> responseObserver) {

        StatusMessage message = buildMessage(
                Status.STATUS_RUNNING,
                "Receiving data",
                LogLevel.LOGLEVEL_INFO);
        responseObserver.onNext(message);

        //1. Read data
        List<TimeSeries> rawWords = data.getInputsList();
        List<TimeSeries> rawOutputs = data.getOutputsList();

        //2. Transform Data to Words
        List<Pair<Word<Integer>, Word<Integer>>> wordObs;
        try {
            wordObs = exportDataRowAsWords(
                    rawWords,
                    rawOutputs,
                    true);
        } catch (MessageException e) {
            responseObserver.onNext(e.getStatusMessage());
            responseObserver.onCompleted();
            return;
        }

        //3. Extract input Alphabet
        Alphabet<Integer> inputAlphabet = extractAlphabet(wordObs);

        message = buildMessage(
                Status.STATUS_RUNNING,
                "Learning Mealy machine",
                LogLevel.LOGLEVEL_INFO);
        responseObserver.onNext(message);

        // 4. Learn Model
        final PassiveMealyLearner<Integer, Integer> learner = new BlueFringeRPNIMealy<>(inputAlphabet);
        for (Pair<Word<Integer>, Word<Integer>> pair : wordObs) {
            learner.addSample(pair.getValue0(), pair.getValue1());
        }

        model = learner.computeModel();

        //TODO: optional artefact Observation table or dot file (see LearnLib Examples)
        //new ObservationTableASCIIWriter<>().write(lstar.getObservationTable(), System.out);
        //Desktop.getDesktop().browse(OTUtils.writeHTMLToFile(lstar.getObservationTable()).toURI());
        //GraphDOT.write(result, driver.getInputs(), System.out);

        message = buildMessage(Status.STATUS_FINISHED, "Model learned successfully", LogLevel.LOGLEVEL_INFO);
        responseObserver.onNext(message);
        responseObserver.onCompleted();
    }

    /**
     * Predicts the final output of a trace
     * 1. Read Data
     * 2. Transform Data to Words
     * 3. Compute Prediction
     *
     * @param data             GRPC DataPackage, rows are traces where entries are alternating inputs and outputs
     * @param responseObserver Interface to transmit status messages via the GRPC connection
     */
    public void predict(io.flowcean.learner.grpc.LearnerOuterClass.DataPackage data,
                        io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Prediction> responseObserver) {
        //1. Read Data
        List<TimeSeries> rawInputs = data.getInputsList();
        List<TimeSeries> rawOutputs = data.getOutputsList();

        // 2. Transform Data to Words
        List<Pair<Word<Integer>, Word<Integer>>> wordObs;
        Builder prediction = Prediction.newBuilder();
        try {
            wordObs = exportDataRowAsWords(
                    rawInputs,
                    rawOutputs,
                    false);
        } catch (MessageException e) {
            prediction.setStatus(buildMessage(
                    Status.STATUS_FAILED,
                    e.getMessage(),
                    LogLevel.LOGLEVEL_FATAL));
            io.flowcean.learner.grpc.LearnerOuterClass.TimeSeries emptyObs = TimeSeries.newBuilder().build();
            prediction.addPredictions(emptyObs);
            responseObserver.onNext(prediction.build());
            responseObserver.onCompleted();
            return;
        }

        //3. Compute Prediction
        for (Pair<Word<Integer>, Word<Integer>> pair : wordObs) {
            Word<Integer> predictedOutput = model.computeOutput(pair.getValue0());
            List<TimeSample> samples = new ArrayList<>();
            for(int i = 0; i < predictedOutput.size(); i++){
                samples.add(io.flowcean.learner.grpc.LearnerOuterClass.TimeSample.newBuilder()
                        .setTime(i).setValue(
                                LearnerOuterClass.DataField.newBuilder().setInt(predictedOutput.getSymbol(i))).build());
            }
            io.flowcean.learner.grpc.LearnerOuterClass.TimeSeries.Builder obs = TimeSeries.newBuilder().addAllSamples(samples);
            prediction.addPredictions(obs.build());
        }

        responseObserver.onNext(prediction.build());
        responseObserver.onCompleted();
    }

    /**
     * Exports the Java-object to a file
     * TODO: this function will change and implement the learner's "save"-functionality
     *
     * @param request          GRPC request token
     * @param responseObserver GRPC interface for a response
     */
    public void export(io.flowcean.learner.grpc.LearnerOuterClass.Empty request,
                       io.grpc.stub.StreamObserver<io.flowcean.learner.grpc.LearnerOuterClass.Empty> responseObserver) {
        try {
            FileOutputStream fileOut = new FileOutputStream("model.ser");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(model);
            out.close();
            fileOut.close();
        } catch (IOException e) {
            responseObserver.onCompleted();
            throw new RuntimeException(e);
        }
        responseObserver.onCompleted();
    }

    /**
     * Exports the GRPCPackage's data rows as input-output pairs of LearnLib's words
     *
     * @param rawInputs  Data from GRPC Package, contains all inputs sequences
     * @param rawOutputs Data from GRPC package, contains all output sequences
     * @return a list of input-output pairs representing observed traces
     * @throws MessageException exception for unexpected data
     */
    List<Pair<Word<Integer>, Word<Integer>>> exportDataRowAsWords(List<TimeSeries> rawInputs, List<TimeSeries> rawOutputs, boolean withOutputs) throws MessageException {
        if (rawInputs.isEmpty()) {
            throw new MessageException("No inputs received");
        }
        if (withOutputs && rawInputs.size() != rawOutputs.size()){
            throw new MessageException("Inputs and outputs have unequal length");
        }

        List<Pair<Word<Integer>, Word<Integer>>> wordObs = new ArrayList<>();

        for (int i = 0; i < rawInputs.size(); i++) {
            List<Integer> inputWord = new ArrayList<>();
            List<Integer> outputWord = new ArrayList<>();

            List<TimeSample> inputSamples = new ArrayList<TimeSample>(rawInputs.get(i).getSamplesList());
            inputSamples.sort((sample1, sample2) -> {return Double.compare(sample1.getTime(),sample2.getTime());});
            List<TimeSample> outputSamples = new ArrayList<TimeSample>();
            if(withOutputs) {
                outputSamples = new ArrayList<TimeSample>(rawOutputs.get(i).getSamplesList());
                outputSamples.sort((sample1, sample2) -> {
                    return Double.compare(sample1.getTime(), sample2.getTime());
                });
                if (inputSamples.size() != outputSamples.size()) {
                    throw new MessageException("Input and output have unequal length");
                }
            }
            for (int j = 0; j < inputSamples.size(); j++) {
                inputWord.add(inputSamples.get(j).getValue().getInt());
                if(withOutputs) {
                    outputWord.add(outputSamples.get(j).getValue().getInt());
                }
            }
            wordObs.add(new Pair<>(Word.fromList(inputWord), Word.fromList(outputWord)));
        }
        return wordObs;
    }

    /**
     * Extracts the input alphabet from the observed data
     *
     * @param wordObs the observed data
     * @return the estimated input alphabet
     */
    Alphabet<Integer> extractAlphabet(List<Pair<Word<Integer>, Word<Integer>>> wordObs) {
        Set<Integer> inputSet = new HashSet<>();

        for (Pair<Word<Integer>, Word<Integer>> observation : wordObs) {
            for (int symbol : observation.getValue0()) {
                inputSet.add(symbol);
            }
        }

        return Alphabets.fromCollection(inputSet);
    }

    /**
     * Abbreviates the built of a StatusMessage
     *
     * @param status  the status type
     * @param message the message of the status
     * @param level   the log level
     * @return status message
     */
    StatusMessage buildMessage(Status status, String message, LogLevel level) {
        return StatusMessage.newBuilder().setStatus(status).addMessages(Message.newBuilder().setMessage(message).setLogLevel(level))
                .build();
    }

}
