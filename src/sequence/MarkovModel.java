package sequence;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class MarkovModel{

	// use this flag if using real MSA with Selex format
	final private static boolean SELEX = true;
	
	// use for forward backwards smoothing, but don't
	// because the transitions are firmly one-way, which
	// means some states will end up with 0 probability
	// when you use smoothing
	final private static boolean SMOOTH = false;
	
	final static Charset ENCODING = StandardCharsets.UTF_8;
	private List<String> sequences;
	private TransitionModel t;
	private EmissionModel e;
	private int numPositions, numEmissions, emissionType;
	private double[] startVector;
	private boolean[] backbone;
	
	public MarkovModel(String dataFileName, int emissionType){
		this.sequences = readFile(dataFileName);
		this.emissionType = emissionType;
		this.numPositions = this.sequences.get(0).length();
		
		// amino acid emissions
		if(emissionType == 0){
			this.numEmissions = 20;
		}
		// nucleotide emissions
		else{
			this.numEmissions = 4;
		}
		
		trainModel();
	}
	
	public static void main(String[] args){
		
		MarkovModel m = new MarkovModel("data/selex_data.txt", 0);
		
		//m.fowardBackward("attcatcttgtatctgtctgtgt");
		//m.fowardBackward("GRCERTFLGHEDCVRGLAILSETEFLSCANDASIRRWQ");
		//m.findBestPath("QHCVETHVDHQGEIWAMCVDDTAKRCLTAGTGSDVKVE");
		m.findBestPath("GRCERTFLGHEDCVRGLAILSETEFLSCANDASIRRWQ");
	}
	
	// Uses Viterbi to find the best path
	public void findBestPath(String querySequence){
		
		char emissionChar;
		int emission;
		
		// initialize the information heavy vector with the original start vector
		Pair<ArrayList<Pair<Integer, Double>>, Double>[] previous = new Pair[startVector.length];
		for(int i = 0; i < startVector.length; i++){
			previous[i] = new Pair(new ArrayList<Pair<Integer, Double>>(), startVector[i]);
		}
		
		for(int i = 1; i < querySequence.length(); i++){
			
			// get the current emission from the sequence
			emissionChar = querySequence.charAt(i);
			emission = getIndex(emissionChar);
			
			// use forward algorithm to generate probabilities
			previous = viterbi(previous, emission);
		}
		
		Pair<ArrayList<Pair<Integer, Double>>, Double> best = printInformation(new double[2], previous, '.', 0, true, querySequence);
		
		constructAlignment(querySequence, best.getKey());
	
	}
	
	private void constructAlignment(String querySequence, List<Pair<Integer, Double>> path){
		StringBuilder aligned = new StringBuilder();
		int sequenceIndex = 0;
		boolean last = true;
		
		// loop through the querySequence, extending if you add a gap
		int limit = querySequence.length();
		for(int i = 0; i < limit; i++){
			if(i < backbone.length && backbone[i] && sequenceIndex < querySequence.length()){
				if(last && path.get(sequenceIndex).getKey() % 3 == 1){
					aligned.append('-');
					limit++;
					last = false;
				}
				else{
					aligned.append(querySequence.charAt(sequenceIndex));
					last = true;
					sequenceIndex++;
				}
			}
			else{
				if(path.get(sequenceIndex).getKey() % 3 == 2){
					aligned.append(querySequence.charAt(sequenceIndex));
					sequenceIndex++;
				}
				else{
					aligned.append('-');
					limit++;
				}
			}
		}
		System.out.println("\nAligned Sequence: " + aligned);
	}
		
	private Pair<ArrayList<Pair<Integer, Double>>, Double>[] viterbi(Pair<ArrayList<Pair<Integer, Double>>, Double>[] previous, int emission){
		previous = t.viterbiTransition(previous);
		
		previous = e.viterbiEmission(previous, emission);
		
		return previous;
	}
	
	public void fowardBackward(String querySequence){
		
		double[] previous = startVector;
		double[][] distributions = new double[querySequence.length()][previous.length];
		
		// don't need to transition to start vector, just update with emission model
		char emissionChar = querySequence.charAt(0);
		int emission = getIndex(emissionChar);
		
		previous = e.update(previous, emission);
		printInformation(previous, null, emissionChar, 0, false, null);
		
		// loop through the rest of the emissions (symbols) in the sequence
		for(int i = 1; i < querySequence.length(); i++){
			
			// store the distribution vector for later reference
			distributions[i] = previous;
			
			// get the current emission from the sequence
			emissionChar = querySequence.charAt(i);
			emission = getIndex(emissionChar);
			
			// use forward algorithm to generate probabilities
			previous = forward(previous, emission);
			printInformation(previous, null, emissionChar, i, false, null);
		}
		
		// note that this doesn't work well (at all) since transitions are one-way
		if(SMOOTH){
			// this is a great line of code. recursive? not technically but intuitively
			previous = new double[previous.length];
			Arrays.fill(previous, 1.0);
			
			for(int i = querySequence.length()-1; i > 0; i--){
				
				emissionChar = querySequence.charAt(i);
				emission = getIndex(emissionChar);
				
				previous = backward(previous, emission);
				
				// use the forward message and recently computed backward message
				// to determine the new, smoothed distribution
				distributions[i-1] = smooth(distributions[i-1], previous);
				printInformation(distributions[i-1], null, querySequence.charAt(i-1), i-1, false, null);
			}
		}
	}
	
	private double[] forward(double[] previous, int emission){
		
		// predict step using transition model
		previous = t.predict(previous);
		
		// update step using emission model
		previous = e.update(previous, emission);
		
		return previous;
	}
	
	public double[] backward(double[] previous, int emission){
		
		// determine probability each location at t+1 given reading at t+1
		previous = e.update(previous, emission);
		
		// determine probability of location at t given probability of
		// all the locations at t+1
		previous = t.predict(previous);
		
		return previous;
	}
	
	public double[] smooth(double[] forward, double[] backward){
		double[] result = new double[forward.length];
		double sum = 0.0;
		
		// multiply the elements of the two vectors
		for(int i = 0; i < result.length; i++){
			result[i] = forward[i] * backward[i]; 
			sum += result[i];
		}
		
		// normalize the result
		for(int i = 0; i < result.length; i++){
			result[i] = result[i]/sum;
		}
		
		return result;
	}
	
	private void trainModel(){
		
		// create a position specific frequency matrix in which
		// first index is the position, second index is the emission
		int[][] psfm = createPSFM();
		
		// create the backbone of the HMM based on the number of
		// filled sequences at each position in the alignment
		backbone = createBackbone();
		
		// figure out how many nodes our model will have
		int numNodes = 0;
		for(boolean matchColumn : backbone){
			if(matchColumn)
				numNodes++;
		}
		
		// create an annotated version of the sequences, denoting
		// whether each position is a match (0), delete (1), or 
		// insert (2) in each sequence. note that if it is not
		// a backbone state and has a blank, it is not a valid
		// state so we annotate it with a -1
		int[][] annotated = createAnnotations(backbone);
		
		// propogate the emission and transition models with the data
		t = new TransitionModel(annotated, backbone, numNodes);
		e = new EmissionModel(psfm, backbone, numNodes);
		
		// I need to figure out how to add an initial FIM, and then
		// transition from this state to the inital states in the model
		
		// create the start vector with the initial state distribution
		startVector = generateStartVector(numNodes, backbone, annotated);		
		
	}
	
	private double[] generateStartVector(int numNodes, boolean[] backbone, int[][] annotatedSequences){
		
		// don't forget to leave space for the FIM
		double[] startVector = new double[numNodes*3 + 1];
		Arrays.fill(startVector, 0.0);
		
		double matchTotal = 1.0;
		double deleteTotal = 1.0;
		double insertTotal = 1.0;
	
		for(int[] sequence : annotatedSequences){
			
			// we have to find the first valid position in each sequence
			int startPosition =  0;
			while(sequence[startPosition] < 0){
				startPosition++;
			}
			
			// we've got ourselves a match
			if(sequence[startPosition] == 0){
				matchTotal++;
			}
			
			// we've got ourselves a delete
			if(sequence[startPosition] == 1){
				deleteTotal++;
			}
			// we've got ourselves a insert
			if(sequence[startPosition] == 2){
				insertTotal++;
			}
		}
		
		// we can find the index of the state in the vector as follows:
		// nodeNumber (start at 0) * 3 + stateNumber (match = 0, delete = 1, insert = 2)
		startVector[0] = matchTotal/(sequences.size()+startVector.length);
		startVector[1] = deleteTotal/(sequences.size()+startVector.length);
		startVector[2] = insertTotal/(sequences.size()+startVector.length);
		
		// pseudocount for every possible start state
		for(int i = 3; i < startVector.length; i++){
			startVector[i] = 1/(sequences.size()+startVector.length);
		}
		
		return t.propogateDelete(startVector);
	}
	
	private int[][] createAnnotations(boolean[] backbone){
		int[][] annotated = new int[sequences.size()][numPositions];
		
		for(int sequenceNum = 0; sequenceNum < sequences.size(); sequenceNum++){
			String sequence = sequences.get(sequenceNum);
			
			for(int position = 0; position < numPositions; position++){
				char emission = sequence.charAt(position);
				
				// if this position is a backbone column in the alignment
				if(backbone[position]){
					// check whether it is a blank emission
					if(emission != '-'){
						// it is not a blank, so the position is a match
						annotated[sequenceNum][position] = 0;
					}
					else{
						// it is a blank, so the position is a delete
						annotated[sequenceNum][position] = 1;
					}
				}
				else{
					// if the position is not a blank, then it is inset
					// otherwise we can just ignore it
					if(emission != '-'){
						annotated[sequenceNum][position] = 2;
					}
					else{
						// annotate with -1 to show it is invalid
						annotated[sequenceNum][position] = -1;
					}
				}	
			}
		}
		return annotated;
	}
	
	private boolean[] createBackbone(){
		boolean[] backbone = new boolean[numPositions];
		
		// determine which positions in the alignment are part of the
		// backbone of the profile hmm (not columns in the pssm/psfm)
		for(int position = 0; position < numPositions; position++){
			int filled = 0;
			for(String sequence : sequences){
				// get the symbol at this position in each sequence
				char emission = sequence.charAt(position);
				
				// if the symbol is not a blank space then count it
				if(emission != '-'){
					filled++;
				}
			}
			
			// if the majority of sequences are filled at this position is
			// then this is part of the backbone
			if(filled > (sequences.size() - filled)){
				// mark this column as part of the backbone
				backbone[position] = true;
			}
			else{
				backbone[position] = false;
			}
		}
		
		return backbone;
	}
	
	
	private int[][] createPSFM(){
		
		int[][] psfm = new int[numPositions][numEmissions];
		
		// initialize each emission with a pseudocount of 1 to avoid
		// overfitting the model to the limited dataset
		for(int[] positionColumn : psfm){
			Arrays.fill(positionColumn, 1);
		}
		
		// loop through each sequence
		for(String sequence : sequences){
			// loop through each position in the sequence
			for(int i = 0; i < numPositions; i++){
				char emission = sequence.charAt(i);
				
				// get the index in the column corresponding to this emission
				int index = getIndex(emission);
				// increment the count of this emission at this position in the psfm
				if(index >= 0){
					psfm[i][index]++;
				}
			}
		}
		return psfm;	
	}
	
	private double[][] createPSSM(int[][] psfm){
		
		double[][] pssm = new double[numPositions][numEmissions];
		
		for(double[] positionColumn : pssm){
			// initialize each emission with a probability of 0
			// note that no emission will end up with a probability
			// of 0 because of the pseudocounts we have included
			Arrays.fill(positionColumn, 0.0);
		}
		
		for(int column = 0; column < numPositions; column++){
			int totalEmissions = numEmissions;
			// loop through each emission and add its frequency to the total
			for(int i = 0; i < numEmissions; i++){
				int emissionFrequency = psfm[column][i];
				totalEmissions += emissionFrequency;
			}
			// the probability is the frequency of this emission divided
			// by the total number of emissions in this position
			for(int i = 0; i < numPositions; i++){
				pssm[column][i] = psfm[column][i]/totalEmissions;
			}
		}	
		return pssm;		
	}
	
	private Pair<ArrayList<Pair<Integer, Double>>, Double> printInformation(double[] distribution, Pair<ArrayList<Pair<Integer, Double>>, Double>[] previous, 
			char emission, int position, boolean viterbi, String querySequence){
		if(viterbi){
			double max = 0.0;
			int maxIndex = 0;
			Pair<ArrayList<Pair<Integer, Double>>, Double> best = null;
			for(int i = 0; i < previous.length; i++){
				if(previous[i].getValue() > max){
					max = previous[i].getValue();
					maxIndex = i;
					best = previous[i];
				}
			}
			
			for(int i = 0; i < best.getKey().size(); i++){
				
				Pair<Integer, Double> node = best.getKey().get(i);
				int index = node.getKey();
				
				// stateNumber is match, delete, or insert
				int stateNumber = index % 3;
				
				// nodeNumber is which node in the model we are at
				int nodeNumber = (int) Math.floor(index / 3);
				
				StringBuilder stateName = new StringBuilder();
				
				if(index == previous.length - 1){
					stateName.append("FIM");
				}
				else{
					if(stateNumber == 0){
						stateName.append("M");
					}
					if(stateNumber == 1){
						stateName.append("D");
					}
					if(stateNumber == 2){
						stateName.append("I");
					}
					
					stateName.append(nodeNumber);
				}
				int j = i;
				while(j < best.getKey().size() && stateNumber == 1){
					j++;
					stateName.append(" + ");
					node = best.getKey().get(j);
					index = node.getKey();
					
					// stateNumber is match, delete, or insert
					stateNumber = index % 3;
					
					// nodeNumber is which node in the model we are at
					nodeNumber = (int) Math.floor(index / 3);
					
					if(index == previous.length - 1){
						stateName.append("FIM");
					}
					else{
						if(stateNumber == 0){
							stateName.append("M");
						}
						if(stateNumber == 1){
							stateName.append("D");
						}
						if(stateNumber == 2){
							stateName.append("I");
						}
						
						stateName.append(nodeNumber);
					}
				}
				System.out.println("The most likely state for position " + i + " with emission " + querySequence.charAt(i) + " is: " + stateName + " with probability " + node.getValue());
			}
			return best;
		}
		else{
			double max = 0;
			int maxIndex = 0;
			
			for(int i = 0; i < distribution.length; i++){
				if(distribution[i] > max){
					max = distribution[i];
					maxIndex = i;
				}
			}
			// stateNumber is match, delete, or insert
			int stateNumber = maxIndex % 3;
			
			// nodeNumber is which node in the model we are at
			int nodeNumber = (int) Math.floor(maxIndex / 3);
			
			StringBuilder stateName = new StringBuilder();
			
			if(maxIndex == distribution.length - 1){
				stateName.append("FIM");
			}
			else{
				if(stateNumber == 0){
					stateName.append("M");
				}
				if(stateNumber == 1){
					stateName.append("D");
				}
				if(stateNumber == 2){
					stateName.append("I");
				}
				
				stateName.append(nodeNumber);
			}
			System.out.println("The most likely state for position " + position + " with emission " + emission + " is: " + stateName + " with probability " + max);
		}
		return null;
	}
	
	/* 
	 * This method transforms amino acid chars into an index for emission
	 */
	private int getIndex(char emission){
		
		if(emissionType ==  0){
			// there are 20 amino acids I did not know that
			if(Character.toLowerCase(emission) == 'a'){
				return 0;
			}
			if(Character.toLowerCase(emission) == 'c'){
				return 1;
			}
			if(Character.toLowerCase(emission) == 'd'){
				return 2;
			}
			if(Character.toLowerCase(emission) == 'e'){
				return 3;
			}
			if(Character.toLowerCase(emission) == 'f'){
				return 4;
			}
			if(Character.toLowerCase(emission) == 'g'){
				return 5;
			}
			if(Character.toLowerCase(emission) == 'h'){
				return 6;
			}
			if(Character.toLowerCase(emission) == 'i'){
				return 7;
			}
			if(Character.toLowerCase(emission) == 'k'){
				return 8;
			}
			if(Character.toLowerCase(emission) == 'l'){
				return 9;
			}
			if(Character.toLowerCase(emission) == 'm'){
				return 10;
			}
			if(Character.toLowerCase(emission) == 'n'){
				return 11;
			}
			if(Character.toLowerCase(emission) == 'p'){
				return 12;
			}
			if(Character.toLowerCase(emission) == 'q'){
				return 13;
			}
			if(Character.toLowerCase(emission) == 'r'){
				return 14;
			}
			if(Character.toLowerCase(emission) == 's'){
				return 15;
			}
			if(Character.toLowerCase(emission) == 't'){
				return 16;
			}
			if(Character.toLowerCase(emission) == 'v'){
				return 17;
			}
			if(Character.toLowerCase(emission) == 'w'){
				return 18;
			}
			if(Character.toLowerCase(emission) == 'y'){
				return 19;
			}
		}
		else{
			// otherwise we have nucleotides
			if(Character.toLowerCase(emission) == 'a'){
				return 0;
			}
			if(Character.toLowerCase(emission) == 't'){
				return 1;
			}
			if(Character.toLowerCase(emission) == 'c'){
				return 2;
			}
			if(Character.toLowerCase(emission) == 'g'){
				return 3;
			}
		}
		return -1;
	}
	
	private static List<String> readFile(String fileName){
		try{
			Path path = Paths.get(fileName);
			if(SELEX){
				List<String> raw_data = Files.readAllLines(path);
				return preprocess(raw_data);
			}
			else{
				return Files.readAllLines(path, ENCODING);
			}
		}
		catch(IOException e){
			System.err.println(e);
			return null;
		}
	}
	
	private static List<String> preprocess(List<String> raw_data){
		
		ArrayList<String> processed = new ArrayList<String>();
		
		for(String raw_line : raw_data){
			String[] tokens = raw_line.split("\\s+");
			processed.add(tokens[1]);
		}
		
		return processed;
	}
	
}
