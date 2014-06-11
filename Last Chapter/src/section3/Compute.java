package section3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;
import afc.mira.Algorithm.RuleStatus;
import afc.mira.Mira;
import afc.predictionstatistics.PredictionStats;

public class Compute implements Callable<Results>{
	private int noiseSize;
	private int numOfRepeat;
	private int numOfFolds;
	private Instances instances;
	private int bi;
	private String filename;
	private int subsampleIndex;
	private Map<String, List<Double>> string2AUCList;
	private Map<String, List<Double>> string2ClassifierUsedList;
	private String[] names;
	private Classifier[] classifiers;
	private int maxSamples;
	
	public Compute(int noiseSize, int numOfRepeat, int numOfFolds, Instances instances, int bi, String filename, int subsampleIndex, 
			Map<String, List<Double>> string2AUCList, Map<String, List<Double>> string2ClassifierUsedList,
			String[] names, Classifier[] classifiers, int maxSamples){
		this.noiseSize = noiseSize;
		this.numOfRepeat = numOfRepeat;
		this.numOfFolds = numOfFolds;
		this.instances = instances;
		this.bi = bi;
		this.filename = filename;
		this.subsampleIndex = subsampleIndex;
		this.string2AUCList = string2AUCList;
		this.string2ClassifierUsedList = string2ClassifierUsedList;
		this.names = names;
		this.classifiers = classifiers;
		this.maxSamples = maxSamples;
	}
	
	@Override
	public Results call() throws Exception {
		System.out.println();
		System.out.println("Noise Size: " + noiseSize + "%");
		/*
		 * Add noise
		 */
		runNoise();
		return null;
	}
	
	private void runNoise() throws Exception{
		List<double[]> aucList = new ArrayList<double[]>();
		for(int r = 0; r < numOfRepeat; r++){
			System.out.println(filename + "\t" + bi + "\t" + noiseSize + "\t" + "Repeats: " + r + " / " + numOfRepeat);
			Random rand = new Random(r);   // create seeded number generator
			Instances randData = new Instances(instances);   // create copy of original data
			randData.randomize(rand);
			randData.stratify(numOfFolds);
			List<Integer> trueClassList = new ArrayList<Integer>();
			Map<String, List<Double>> name2PredictionList = new HashMap<String, List<Double>>();
			runCrossValidation(numOfFolds, randData, noiseSize, name2PredictionList, trueClassList, bi);
			double[] auc = new double[names.length];
			for(int i = 0; i < names.length; i++){
				String key = filename + "_" + subsampleIndex + "_" + noiseSize + "_" + bi + "_" + names[i];
				PredictionStats stats = new PredictionStats(trueClassList, name2PredictionList.get(key));
				if(string2AUCList.containsKey(key) == false) string2AUCList.put(key, new ArrayList<Double>());
				double a = stats.computeAUC();
				auc[i] = a;
				string2AUCList.get(key).add(a);
			}
			aucList.add(auc);
		}  
		/*
		 * Checking
		 */
		for(int i = 0; i < names.length; i++){
			for(double[] d:aucList){
				System.out.print(d[i] + ", ");
			}
			System.out.println();
		}
	}
	
	private void runCrossValidation(final int numOfFolds, Instances randData, int noiseSize, 
			Map<String, List<Double>> name2PredictionList, 
			List<Integer> trueClassList, int baggingIterations) throws Exception{
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
			for(int x = 0; x < test.numInstances(); x++){
				trueClassList.add((int)test.instance(x).classValue());
			}
			runPrediction(train, test, name2PredictionList, baggingIterations);
		}
	}
	
	private void runPrediction(Instances train, Instances test, 
			Map<String, List<Double>> name2PredictionList, int baggingIterations) throws Exception{
		/*
		 * Returns the number of iterations ran for each instance
		 * 
		 * Classifier score could be one of two methods
		 * 		1) Use average of all classifiers - default (the one mentioned in thesis)
		 * 		2) total classifiers predicting positive / total number of classifiers used
		 */
		if(baggingIterations != -1){
			/*
			 * Typical bagging with fixed iterations
			 */
			/*
			 * Training
			 */
			Map<String, List<Classifier>> name2ClassifierList = new HashMap<String, List<Classifier>>();
			for(int i = 0; i < names.length; i++){
				List<Classifier> cList = new ArrayList<Classifier>();
				for(int x = 0; x < baggingIterations; x++){
					if(x != 0 && x % 100 == 0) System.out.println(x + " / " + baggingIterations);
					Classifier c = Classifier.makeCopy(classifiers[i]);
					Instances tempInstances = getBootstrap(train, x);
					c.buildClassifier(tempInstances);
					cList.add(c);
				}
				name2ClassifierList.put(names[i], cList);
			}
			/*
			 * Predicting
			 */
			for(int i = 0; i < names.length; i++){
				List<Double> predictionList = new ArrayList<Double>();
				double totalSamples = 0;
				for(int x = 0; x < test.numInstances(); x++){
					double score = 0.0;
					for(int y = 0; y < name2ClassifierList.get(names[i]).size(); y++){
						score += name2ClassifierList.get(names[i]).get(y).distributionForInstance(test.instance(x))[0];
					}
					score /= baggingIterations;
					predictionList.add(score);
					totalSamples += baggingIterations;
				}
				totalSamples /= test.numInstances();
				String key = filename + "_" + subsampleIndex + "_" + noiseSize + "_" + bi + "_" + names[i];
				if(string2ClassifierUsedList.containsKey(key) == false) string2ClassifierUsedList.put(key, new ArrayList<Double>());
				if(name2PredictionList.containsKey(key) == false) name2PredictionList.put(key, new ArrayList<Double>());
				string2ClassifierUsedList.get(key).add(totalSamples);
				name2PredictionList.get(key).addAll(predictionList);
			}
		}else{
			/*
			 * Run Mira
			 */
			for(int i = 0; i < names.length; i++){
				List<Double> predictionList = new ArrayList<Double>();
				double meanSamplesUsedList = runMira(train, test, classifiers[i], predictionList);
				
				String key = filename + "_" + subsampleIndex + "_" + noiseSize + "_" + bi + "_" + names[i];
				if(string2ClassifierUsedList.containsKey(key) == false) string2ClassifierUsedList.put(key, new ArrayList<Double>());
				if(name2PredictionList.containsKey(key) == false) name2PredictionList.put(key, new ArrayList<Double>());
				string2ClassifierUsedList.get(key).add(meanSamplesUsedList);
				name2PredictionList.get(key).addAll(predictionList);
			}
		}
	}
	
	private double runMira(Instances train, Instances test, Classifier baseClassifier, List<Double> predictionList) throws Exception{
		final double theta = 0.5;
		final double alpha = 0.0001;
		final double beta = 0.0001;
		double totalSamples = 0;
		List<Classifier> trainedClassifierList = new ArrayList<Classifier>();
		for(int x = 0; x < test.numInstances(); x++){
			double totalScore = 0.0;
			Mira trainMira = new Mira(theta, alpha, beta, maxSamples);
			//Generate new train classifiers until a conclusion could be made
			int counter = 0;
			while(trainMira.obtainAnotherSample()){
				if(counter >= trainedClassifierList.size()){
					//Train Another Classifier
					Classifier cloneC = Classifier.makeCopy(baseClassifier);
					cloneC.buildClassifier(this.getBootstrap(train, counter));
					trainedClassifierList.add(cloneC);
				}
				double score = trainedClassifierList.get(counter).distributionForInstance(test.instance(x))[0];
				totalScore += score;
				if(score > 0.5) trainMira.update(true, RuleStatus.APNOTSATISFIED);
				else trainMira.update(false, RuleStatus.APNOTSATISFIED);
				counter++;
			}
			System.out.print(counter + ",");
			predictionList.add((totalScore/counter));
			totalSamples += counter;
		}
		System.out.println();
		return totalSamples /= test.numInstances();
	}
	
	private Instances getBootstrap(Instances inst, int seed) throws Exception{
		Resample filter = new Resample();
		filter.setInputFormat(inst);
		filter.setRandomSeed(seed);
		return Filter.useFilter(inst, filter);
	}
}
