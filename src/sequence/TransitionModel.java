package sequence;

import java.util.ArrayList;
import java.util.Arrays;

class TransitionModel{
	State[][] stateTransitions;
	
	public TransitionModel(int[][] annotatedSequences, boolean[] backbone, int numNodes){
		
		// note we have more nodes than the number of insert/match pairs
		// this is because we have a free insertion module at the end
		// (also note that because we assume insert states precede match
		// states, the first insert is actually a FIM as well)
		this.stateTransitions = new State[3][numNodes+1];
		
		this.stateTransitions[0] = new State[numNodes+1];
		this.stateTransitions[1] = new State[numNodes+1];
		this.stateTransitions[2] = new State[numNodes+1];
		
		createTransitions(annotatedSequences, backbone);
	}

	public double[] predict(double[] previous){
		// remember, this is how you index into the distribution vector:
		// nodeNumber (start at 0) * 3 + stateNumber (match = 0, delete = 1, insert = 2)
		
		double[] current = new double[previous.length];
		Arrays.fill(current, 0.0);
		
		// don't look at last state (FIM) because it always transitions to itself
		for(int index = 0; index < previous.length - 1; index++){
			// the probability of being in a particular state
			double probability = previous[index];
			
			// stateNumber is match, delete, or insert
			int stateNumber = index % 3;
			
			// nodeNumber is which node in the model we are at
			int nodeNumber = (int) Math.floor(index / 3);
			
			// find the probability of being in any states next
			// turn based on the transition probabilities from this
			// state multiplied by the probability of this state
			
			State transitions = stateTransitions[stateNumber][nodeNumber];
			
			// we will find the index of the states we are transitioning to
			int nextMatch, nextDelete, nextInsert;
			
			// next state is in the current node if you are an insert state
			if(stateNumber == 2){
				nextMatch = index - 2;
				nextDelete = index - 1;
				nextInsert = index;
			}
			else{
				
				// index of next state is in next node for match/delete
				nextMatch = (nodeNumber + 1)  * 3;
				nextDelete = (nodeNumber + 1)  * 3 + 1;		
				nextInsert = (nodeNumber + 1)  * 3 + 2;
			}
			
			if(nextMatch < current.length){
				// probability of transition to next match state
				current[nextMatch] += transitions.getToMatch() * probability;
			}
			if(nextDelete < current.length){
				// probability of transition to next insert state
				current[nextDelete] += transitions.getToDelete() * probability;
			}
			if(nextInsert < current.length){
				// probability of transition to next insert state
				current[nextInsert] += transitions.getToInsert() * probability;
			}
			// probability of transition to FIM
			current[current.length - 1] += transitions.getToFIM() * probability;
		}
		
		// now you have to make sure probability of being in delete
		// state is distributed amongst emitting states it transitions to
		
		return propogateDelete(current);
	}
	
	public Pair<ArrayList<Pair<Integer, Double>>, Double>[] viterbiTransition(Pair<ArrayList<Pair<Integer, Double>>, Double>[] previous){
		// remember, this is how you index into the distribution vector:
		// nodeNumber (start at 0) * 3 + stateNumber (match = 0, delete = 1, insert = 2)
		
		Pair<ArrayList<Pair<Integer, Double>>, Double>[] current = new Pair[previous.length];
		for(int i = 0; i < current.length; i++){
			current[i] = new Pair<ArrayList<Pair<Integer, Double>>, Double>(new ArrayList<Pair<Integer, Double>>(), 0.0);
		}
		
		// don't look at last state (FIM) because it always transitions to itself
		for(int index = 0; index < previous.length - 1; index++){
			// the probability of being in a particular state
			double probability = previous[index].getValue();
			
			// stateNumber is match, delete, or insert
			int stateNumber = index % 3;
			
			// nodeNumber is which node in the model we are at
			int nodeNumber = (int) Math.floor(index / 3);
			
			// find the probability of being in any states next
			// turn based on the transition probabilities from this
			// state multiplied by the probability of this state
			
			State transitions = stateTransitions[stateNumber][nodeNumber];
			
			// we will find the index of the states we are transitioning to
			int nextMatch, nextDelete, nextInsert;
			
			// next state is in the current node if you are an insert state
			if(stateNumber == 2){
				nextMatch = index - 2;
				nextDelete = index - 1;
				nextInsert = index;
			}
			else{
				// index of next state is in next node for match/delete
				nextMatch = (nodeNumber + 1)  * 3;
				nextDelete = (nodeNumber + 1)  * 3 + 1;		
				nextInsert = (nodeNumber + 1)  * 3 + 2;
			}
			
			if(nextMatch < current.length){
				// max probability of transition to next match state
				if(transitions.getToMatch() * probability > current[nextMatch].getValue()){
					ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
					backpointer.addAll(previous[index].getKey());
					backpointer.add(new Pair(index, probability));
					current[nextMatch].setKey(backpointer);
					current[nextMatch].setValue(transitions.getToMatch() * probability);
				}
			}
			if(nextDelete < current.length){
				// max probability of transition to next insert state
				if(transitions.getToDelete() * probability > current[nextDelete].getValue()){
					ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
					backpointer.addAll(previous[index].getKey());
					backpointer.add(new Pair(index, probability));
					current[nextDelete].setKey(backpointer);
					current[nextDelete].setValue(transitions.getToDelete() * probability);
				}
			}
			if(nextInsert < current.length){
				// probability of transition to next insert state
				if(transitions.getToInsert() * probability > current[nextInsert].getValue()){
					ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
					backpointer.addAll(previous[index].getKey());
					backpointer.add(new Pair(index, probability));
					current[nextInsert].setKey(backpointer);
					current[nextInsert].setValue(transitions.getToInsert() * probability);
				}
			}
			// probability of transition to FIM
			if(transitions.getToFIM() * probability > current[current.length - 1].getValue()){
				ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
				backpointer.addAll(previous[index].getKey());
				backpointer.add(new Pair(index, probability));
				current[current.length - 1].setKey(backpointer);
				current[current.length - 1].setValue(transitions.getToFIM() * probability);
			}
		}
		
		// now you have to make sure probability of being in delete
		// state is distributed amongst emitting states it transitions to
		
		return propogateViterbi(current);
	}
	
	public Pair<ArrayList<Pair<Integer, Double>>, Double>[] propogateViterbi(Pair<ArrayList<Pair<Integer, Double>>, Double>[] current){
		
		for(int index = 0; index < current.length; index++){
			// stateNumber is match, delete, or insert
			int stateNumber = index % 3;
			
			// nodeNumber is which node in the model we are at
			int nodeNumber = (int) Math.floor(index / 3);
			
			// if we are at a delete state
			if(stateNumber == 1){
				
				double probability = current[index].getValue();
				
				// we must distribute probabilities to future match/insert
				// states so that they receive the current emission				
				if(probability != 0){	
					
					State transitions = stateTransitions[stateNumber][nodeNumber];
					
					// get indices of the next states
					int nextMatch = (nodeNumber + 1)  * 3;
					int nextDelete = (nodeNumber + 1)  * 3 + 1;
					int nextInsert = (nodeNumber + 1)  * 3 + 2;

					// distribute the probabilities amongst successors
					if(nextMatch < current.length){
						// max probability of transition to next match state
						if(transitions.getToMatch() * probability > current[nextMatch].getValue()){
							ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
							backpointer.addAll(current[index].getKey());
							backpointer.add(new Pair(index, probability));
							current[nextMatch].setKey(backpointer);
							current[nextMatch].setValue(transitions.getToMatch() * probability);
						}
					}
					if(nextDelete < current.length){
						// max probability of transition to next insert state
						if(transitions.getToDelete() * probability > current[nextDelete].getValue()){
							ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
							backpointer.addAll(current[index].getKey());
							backpointer.add(new Pair(index, probability));
							current[nextDelete].setKey(backpointer);
							current[nextDelete].setValue(transitions.getToDelete() * probability);
						}
					}
					if(nextInsert < current.length){
						// probability of transition to next insert state
						if(transitions.getToInsert() * probability > current[nextInsert].getValue()){
							ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
							backpointer.addAll(current[index].getKey());
							backpointer.add(new Pair(index, probability));
							current[nextInsert].getKey();
							current[nextInsert].setValue(transitions.getToInsert() * probability);
						}
					}
					// probability of transition to FIM
					if(transitions.getToFIM() * probability > current[current.length - 1].getValue()){
						ArrayList<Pair<Integer, Double>> backpointer = new ArrayList<Pair<Integer, Double>>();
						backpointer.addAll(current[index].getKey());
						backpointer.add(new Pair(index, probability));
						current[current.length - 1].setKey(backpointer);
						current[current.length - 1].setValue(transitions.getToFIM() * probability);
					}
					
					// now the probability has been distributed, so remove it from delete
					current[index].setValue(0.0);
				}
			}
		}
		return current;
	}
	
	public double[] propogateDelete(double[] current){
		for(int index = 0; index < current.length; index++){
			// stateNumber is match, delete, or insert
			int stateNumber = index % 3;
			
			// nodeNumber is which node in the model we are at
			int nodeNumber = (int) Math.floor(index / 3);
			
			// if we are at a delete state
			if(stateNumber == 1){
				
				double probability = current[index];
				
				// we must distribute probabilities to future match/insert
				// states so that they receive the current emission				
				if(probability != 0){	
					
					State transitions = stateTransitions[stateNumber][nodeNumber];
					
					// get indices of the next states
					int nextMatch = (nodeNumber + 1)  * 3;
					int nextDelete = (nodeNumber + 1)  * 3 + 1;
					int nextInsert = (nodeNumber + 1)  * 3 + 2;

					// distribute the probabilities amongst successors
					if(nextMatch < current.length){
						// probability of transition to next match state
						current[nextMatch] += transitions.getToMatch() * probability;
					}
					if(nextDelete < current.length){
						// probability of transition to next insert state
						current[nextDelete] += transitions.getToDelete() * probability;
					}
					if(nextInsert < current.length){
						// probability of transition to next insert state
						current[nextInsert] += transitions.getToInsert() * probability;
					}
					current[current.length - 1] += transitions.getToFIM() * probability;
					
					// now the probability has been distributed, so remove it from delete
					current[index] = 0;
				}
			}
		}
		return current;
	}
	
	private void createTransitions(int[][] annotatedSequences, boolean[] backbone){
		
		double matchTotal, deleteTotal, insertTotal;
		int matchToMatch, matchToInsert, matchToDelete, insertToMatch, insertToInsert,
		insertToDelete, deleteToMatch, deleteToInsert, deleteToDelete, toFIM;
		
		int nodeNumber = 0;
		
		State[] matchStates = stateTransitions[0];
		State[] deleteStates = stateTransitions[1];
		State[] insertStates = stateTransitions[2];
		
		// loop through every position in our multiple sequence alignment
		// note that these are not the indices of the states (we call those nodes)
		for(int position = 0; position < backbone.length; position++){
			
			matchTotal = 3;
			matchToMatch = 1;
			matchToDelete = 1;
			matchToInsert = 1;
			
			insertTotal = 3;
			insertToMatch = 1;
			insertToDelete = 1;
			insertToInsert = 1;
			
			deleteTotal = 3;
			deleteToMatch = 1;
			deleteToDelete = 1;
			deleteToInsert = 1;
			
			toFIM = 0;
			
			// while we keep seeing insert columns
			while(position < backbone.length && !backbone[position]){
				for(int[] sequence : annotatedSequences){
					
					// we have to find the next valid position that is not -1
					int nextPosition = position + 1;
					while(nextPosition < sequence.length && sequence[nextPosition] < 0){
						nextPosition++;
					}
					
					if(nextPosition == sequence.length){
						toFIM = 1;
					}
					else{
						// we've got ourselves a insert
						if(sequence[position] == 2){
							insertTotal++;
							if(sequence[nextPosition] == 0){
								// insert->match
								insertToMatch++;
							}
							if(sequence[nextPosition] == 1){
								// insert->delete
								insertToDelete++;
							}
							if(sequence[nextPosition] == 2){
								// insert->insert
								insertToInsert++;
							}
						}
					}
				}
				// loop to the next position in the sequences
				position++;
			}
			
			// if we reached FIM from the insert state, then we will reach it here
			// so don't even enter the loop to save some time
			if(toFIM != 1){
				// we exited the loop, so we must be at a match state. get the
				// transitions for deletes and matches
				for(int[] sequence : annotatedSequences){
					
					// we have to find the next valid position that is not -1
					int nextPosition = position + 1;
					while(nextPosition < sequence.length && sequence[nextPosition] < 0){
						nextPosition++;
					}
					
					// if we get to an FIM from a backbone column, same procedure as before
					if(nextPosition == sequence.length){
						toFIM = 1;
					}
					else{
						// we've got ourselves a match
						if(sequence[position] == 0){
							matchTotal++;
							
							if(sequence[nextPosition] == 0){
								// match->match
								matchToMatch++;
							}
							if(sequence[nextPosition] == 1){
								// match->delete
								matchToDelete++;
							}
							if(sequence[nextPosition] == 2){
								// match->insert
								matchToInsert++;
							}
						}
						
						// we've got ourselves a delete
						if(sequence[position] == 1){
							deleteTotal++;
							if(sequence[nextPosition] == 0){
								// delete->match
								deleteToMatch++;
							}
							if(sequence[nextPosition] == 1){
								// delete->delete
								deleteToDelete++;
							}
							if(sequence[nextPosition] == 2){
								// delete->insert
								deleteToInsert++;
							}
						}
					}
				}
			}
			
			// now you have all the totals, and the frequencies of
			// each transition type, so determine probabilities
			// make sure you handle the case when you have no transitions
			if(matchTotal != 0){
				matchStates[nodeNumber] = new State(matchToMatch/matchTotal, matchToDelete/matchTotal, matchToInsert/matchTotal, toFIM);
			}
			else{
				matchStates[nodeNumber] = new State(0, 0, 0, toFIM);
			}
			if(deleteTotal != 0){
				deleteStates[nodeNumber] = new State(deleteToMatch/deleteTotal, deleteToDelete/deleteTotal, deleteToInsert/deleteTotal, toFIM);
			}
			else{
				deleteStates[nodeNumber] = new State(0, 0, 0, toFIM);
			}
			if(insertTotal != 0){
				insertStates[nodeNumber] = new State(insertToMatch/insertTotal, insertToDelete/insertTotal, insertToInsert/insertTotal, toFIM);
			}
			else{
				insertStates[nodeNumber] = new State(0, 0, 0, toFIM);
			}
			
			// now we can move on to the next node
			nodeNumber++;
		}
	}
	
	private class State{
		private double toMatch, toDelete, toInsert, toFIM;
		
		private State(double toMatch, double toDelete, double toInsert, double toFIM){
			this.toMatch = toMatch;
			this.toDelete = toDelete;
			this.toInsert = toInsert;
			this.toFIM = toFIM;
		}
		
		private double getToMatch(){
			return toMatch;
		}
		private double getToDelete(){
			return toDelete;
		}
		private double getToInsert(){
			return toInsert;
		}
		private double getToFIM(){
			return toFIM;
		}
	}
}
