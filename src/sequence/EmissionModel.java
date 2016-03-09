package sequence;

import java.util.ArrayList;
import java.util.Arrays;

class EmissionModel{
	
	// these are each a list of emissions lists
	double[][] matchEmissionsList, insertEmissionsList;
	
	public EmissionModel(int[][] psfm, boolean[] backbone, int numNodes){
		
		// we store FIM emission probabilities with match for ease
		matchEmissionsList = new double[numNodes + 1][psfm[0].length];
		insertEmissionsList = new double[numNodes][psfm[0].length];
		
		generateEmissions(psfm, backbone);
	}
	
	public double[] update(double[] previous, int emissionIndex){
		
		double[] current = new double[previous.length];
		Arrays.fill(current, 0.0);
		double sum = 0;
		
		for(int index = 0; index < previous.length; index++){
			
			// stateNumber is match, delete, or insert
			int stateNumber = index % 3;
			
			// nodeNumber is which node in the model we are at
			int nodeNumber = (int) Math.floor(index / 3);
			
			double[][] emissionsList;
			// set the emissions list based on the state
			if(stateNumber == 0){
				emissionsList = matchEmissionsList;
			}
			else if(stateNumber == 2){
				emissionsList = insertEmissionsList;
			}
			else{
				// skip the delete states, they don't matter
				continue;
			}
			
			// now we can get the emission probability based on the node number
			// and which emission was observed by the model
			double emissionProbability = emissionsList[nodeNumber][emissionIndex];
			
			// set the current probability of this state based on the emission
			// and the probability of being in this state in the previous vector
			current[index] += emissionProbability * previous[index];
			sum += current[index];
		}
		return normalize(current, sum);
	}
	
	public Pair<ArrayList<Pair<Integer, Double>>, Double>[] viterbiEmission(Pair<ArrayList<Pair<Integer, Double>>, Double>[] previous, int emissionIndex){
		
		Pair<ArrayList<Pair<Integer, Double>>, Double>[] current = new Pair[previous.length];
		for(int i = 0; i < current.length; i++){
			current[i] = new Pair(new ArrayList<Pair<Integer, Double>>(), 0.0);
		}
		
		for(int index = 0; index < previous.length; index++){
			
			// stateNumber is match, delete, or insert
			int stateNumber = index % 3;
			
			// nodeNumber is which node in the model we are at
			int nodeNumber = (int) Math.floor(index / 3);
			
			double[][] emissionsList;
			// set the emissions list based on the state
			if(stateNumber == 0){
				emissionsList = matchEmissionsList;
			}
			else if(stateNumber == 2){
				emissionsList = insertEmissionsList;
			}
			else{
				// skip the delete states, they don't matter
				continue;
			}
			
			// now we can get the emission probability based on the node number
			// and which emission was observed by the model
			double emissionProbability = emissionsList[nodeNumber][emissionIndex];
			
			// set the current probability of this state based on the emission
			// and the probability of being in this state in the previous vector
			
			current[index].setValue(emissionProbability * previous[index].getValue());
			current[index].setKey(previous[index].getKey());
		}
		
		return current;
	}
	
	private double[] normalize(double[] current, double sum){
		for(int i = 0; i < current.length; i++){
			current[i] = current[i] / sum;
		}
		
		return current;
	}
	
	private void generateEmissions(int[][] psfm, boolean[] backbone){
		int[] insertEmissions, matchEmissions;
		double totalInsertEmissions, totalMatchEmissions;
		
		int nodeNumber = 0;
		
		for(int position = 0; position < backbone.length; position++){
			insertEmissions = new int[psfm[0].length];
			matchEmissions = new int[psfm[0].length];
			totalInsertEmissions = 0.0;
			totalMatchEmissions = 0.0;
			
			while(position < backbone.length && !backbone[position]){
				// we are in an insert state, keep gathering data until
				// about transitions from inserts until we hit a match
				for(int emissionIndex = 0; emissionIndex < psfm[position].length; emissionIndex++){
					insertEmissions[emissionIndex] += psfm[position][emissionIndex];
					totalInsertEmissions += insertEmissions[emissionIndex];
				}
				// continue to the next position in the sequences
				position++;
			}
			
			// we exited the while loop so we must have hit a backbone column
			// so gather information about match emission frequencies
			for(int emissionIndex = 0; emissionIndex < psfm[position].length; emissionIndex++){
				matchEmissions[emissionIndex] += psfm[position][emissionIndex];
				totalMatchEmissions += matchEmissions[emissionIndex];
			}
			
			// now we can calculate our emission probabilities, note that
			// pseudocounts were used in propogating the psfm so we are not overfitting
			for(int emissionIndex = 0; emissionIndex < psfm[position].length; emissionIndex++){
				if(totalInsertEmissions != 0){
					insertEmissionsList[nodeNumber][emissionIndex] = insertEmissions[emissionIndex]/totalInsertEmissions;
				}
				else{
					insertEmissionsList[nodeNumber][emissionIndex] = 0;
				}
				if(totalMatchEmissions != 0){
					matchEmissionsList[nodeNumber][emissionIndex] = matchEmissions[emissionIndex]/totalMatchEmissions;
				}
				else{
					matchEmissionsList[nodeNumber][emissionIndex] = 0;
				}
			}
			
			// now we can move on to the next node
			nodeNumber++;
			
		}
		
		// every emission has equal probability in the FIM
		for(int i = 0; i < matchEmissionsList[nodeNumber].length; i++){
			matchEmissionsList[nodeNumber][i] = 1.0 / matchEmissionsList[nodeNumber].length;
		}
		
	}
	
}