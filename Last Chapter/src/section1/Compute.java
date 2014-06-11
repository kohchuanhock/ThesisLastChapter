package section1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import section3.Results;
import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instances;
import afc.basic.statistics.Mean;
import afc.predictionstatistics.PredictionStats;

public class Compute implements Callable<Results>{
	private int numOfRepeat;
	private int numOfFolds;
	private Instances instances;
	private String[] names;
	private Classifier[] classifiers;
	private int noiseSize;
	private Map<String, List<Double>> string2AUCList;
	private Map<String, List<Double>> string2MeanAUCList;
	private String filename;
	private int subsampleIndex;
	
	public Compute(final int numOfRepeat, final int numOfFolds, Instances instances, String[] names, Classifier[] classifiers,
			int noiseSize, String filename, int subsampleIndex, Map<String, List<Double>> string2AUCList, Map<String, List<Double>> string2MeanAUCList){
		this.numOfRepeat = numOfRepeat;
		this.numOfFolds = numOfFolds;
		this.instances = instances;
		this.names = names;
		this.classifiers = classifiers;
		this.noiseSize = noiseSize;
		this.filename = filename;
		this.subsampleIndex = subsampleIndex;
		this.string2AUCList = string2AUCList;
		this.string2MeanAUCList = string2MeanAUCList;
	}
	
	@Override
	public Results call() throws Exception {
		System.out.println();
		System.out.println("Noise Size: " + noiseSize + "%");
		/*
		 * Add noise
		 */
		Map<String, List<Double>> name2AUCList = runNoise();
		for(int i = 0; i < names.length; i++){
			for(double d:name2AUCList.get(names[i])) System.out.print(d + ", ");
			System.out.println();
			String key = filename + subsampleIndex + noiseSize + names[i];
			if(string2AUCList.containsKey(key)) throw new Error("Duplicate Key");
			string2AUCList.put(key, name2AUCList.get(names[i]));
			key = noiseSize + names[i];
			if(string2MeanAUCList.containsKey(key) == false){
				string2MeanAUCList.put(key, new ArrayList<Double>());
			}
			string2MeanAUCList.get(key).add(Mean.compute(name2AUCList.get(names[i])));
		}
		return null;
	}
	
	private Map<String, List<Double>> runNoise() throws Exception{
		Map<String, List<Double>> name2AUCList = new HashMap<String, List<Double>>();
		for(int r = 0; r < numOfRepeat; r++){
			Random rand = new Random(r);   // create seeded number generator
			Instances randData = new Instances(instances);   // create copy of original data
			randData.randomize(rand);
			randData.stratify(numOfFolds);
			List<Integer> trueClassList = new ArrayList<Integer>();
			Map<String, List<Double>> name2PredictionList = runCrossValidation(numOfFolds, randData, noiseSize, names, classifiers, trueClassList);
			for(int i = 0; i < names.length; i++){
				if(name2AUCList.containsKey(names[i]) == false){
					name2AUCList.put(names[i], new ArrayList<Double>());
				}
				PredictionStats stats = new PredictionStats(trueClassList, name2PredictionList.get(names[i]));
				name2AUCList.get(names[i]).add(stats.computeAUC());
			}
		}  
		return name2AUCList;
	}
	
	private Map<String, List<Double>> runCrossValidation(final int numOfFolds, Instances randData, int noiseSize, String[] names, Classifier[] classifiers,
			List<Integer> trueClassList) throws Exception{
		Map<String, List<Double>> name2PredictionList = new HashMap<String, List<Double>>();
		for (int n = 0; n < numOfFolds; n++) {
			Instances train = randData.trainCV(numOfFolds, n);
			/*
			 * Add noise if needed
			 */
			if(noiseSize > 0){
				int numToAddNoise = (int)Math.round(noiseSize / 100.0 * train.numInstances());
				for(int index = 0; index < numToAddNoise; index++){
					/*
					 * Flip class values
					 */
					double classValue = train.instance(index).classValue();
					if(classValue == 0.0) train.instance(index).setClassValue(1.0);
					else if(classValue == 1.0) train.instance(index).setClassValue(0.0);
					else throw new Error("Neither 0.0 or 1.0: " + train.instance(index).classValue()); 
				}
			}
			Instances test = randData.testCV(numOfFolds, n);
			for(int i = 0; i < test.numInstances(); i++){
				trueClassList.add((int)test.instance(i).classValue());
			}
			for(int i = 0; i < names.length; i++){
				if(name2PredictionList.containsKey(names[i]) == false){
					name2PredictionList.put(names[i], new ArrayList<Double>());
				}
				name2PredictionList.get(names[i]).addAll(runPrediction(train, test, Classifier.makeCopy(classifiers[i])));
			}
		}
		return name2PredictionList;
	}

	private List<Double> runPrediction(Instances train, Instances test, Classifier classifier) throws Exception{
		List<Double> dList = new ArrayList<Double>();
		classifier.buildClassifier(train);
		for(int i = 0; i < test.numInstances(); i++){
			dList.add(classifier.distributionForInstance(test.instance(i))[0]);
			if(test.instance(i).classValue() != 0.0 && test.instance(i).classValue() != 1.0) 
				throw new Error("Neither 0.0 or 1.0: " +  test.instance(i).classValue());
		}
		return dList;
	}
}
