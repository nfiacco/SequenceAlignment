package sequence;

class Pair<T, U>{
		private T key;
		private U value;
		
		public Pair(T key, U value){
			this.key = key;
			this.value = value;
		}
		
		public T getKey(){
			return key;
		}
		
		public U getValue(){
			return value;
		}
		
		public void setKey(T newKey){
			this.key = newKey;
		}
		
		public void setValue(U newValue){
			this.value = newValue;
		}
	}
	