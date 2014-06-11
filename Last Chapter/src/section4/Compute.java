package section4;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import section3.Results;
import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;
import afc.commons.Utils;
import afc.mira.Algorithm.RuleStatus;
import afc.mira.Mira;

public class Compute implements Callable<Results>{
	private int noiseSize;
	private int numOfRepeat;
	private int numOfFolds;
	private Instances instances;
	private int bi[];
	private String filename;
	private String name;
	private Classifier classifier;
	private int maxSamples;
	private double alpha;
	private double beta;

	public Compute(int noiseSize, int numOfRepeat, int numOfFolds, Instances instances, int[] bi, String filename, 
			String name, Classifier classifier, int maxSamples, double errorRates) throws Exception{
		this.noiseSize = noiseSize;
		this.numOfRepeat = numOfRepeat;
		this.numOfFolds = numOfFolds;
		this.instances = instances;
		this.bi = bi;
		this.filename = filename;
		this.name = name;
		this.classifier = classifier;
		this.maxSamples = maxSamples;
		this.alpha = errorRates;
		this.beta = errorRates;
		runNoise();
	}

	@Override
	public Results call() throws Exception {
		runNoise();
		return null;
	}

	private void runNoise() throws Exception{
		int totalCompared = 0;
		int OSMACount = 0;
		int OSMBCount = 0;
		int errorByOSMA = 0;
		int errorByOSMB = 0;
		int errorAt50ByOSMA = 0;
		int errorAt50ByOSMB = 0;
		List<Integer> totalSampleList = new ArrayList<Integer>();
		for(int r = 0; r < numOfRepeat; r++){
			Random rand = new Random(r);   // create seeded number generator
			Instances randData = new Instances(instances);   // create copy of original data
			randData.randomize(rand);
			randData.stratify(numOfFolds);
			List<Double> list0 = runCrossValidation(numOfFolds, randData, noiseSize, bi[0], totalSampleList);
			List<Double> list1 = runCrossValidation(numOfFolds, randData, noiseSize, bi[1], null);
			/*
			 * Check if predictions are same
			 */
			if(list0.size() != list1.size()) throw new Error();
			for(int j = 0; j < list0.size(); j++){
				double largeBag = list1.get(j);
				double mira = list0.get(j);
				
				if(mira != 0.0 && mira != 0.5 && mira != 1.0 && mira != 0.1 && mira != 0.9 && mira != 0.4) throw new Error();
				if(largeBag != 0.0 && largeBag != 0.5 && largeBag != 1.0 && largeBag != 0.1 && largeBag != 0.9) throw new Error();
				
				if(largeBag == 0.5){
					if(mira == 1.0 || mira == 0.0){
						OSMACount++;
						errorAt50ByOSMA++;
					} else if(mira == 0.9 || mira == 0.1){
						OSMBCount++;
						errorAt50ByOSMB++;
					} else if(mira == 0.5) OSMACount++;
					else if(mira == 0.4) OSMBCount++;
					else throw new Error();
				}else if(largeBag == 0.0){
					if(mira == 1.0 || mira == 0.5){
						OSMACount++;
						errorByOSMA++;
					}else if(mira == 0.9 || mira == 0.4){
						OSMBCount++;
						errorByOSMB++;
					}else if(mira == 0.1){
						OSMBCount++;
					}else if(mira == 0.0){
						OSMACount++;
					}else throw new Error();
				}else if(largeBag == 1.0){
					if(mira == 0.0 || mira == 0.5){
						OSMACount++;
						errorByOSMA++;
					}else if(mira == 0.1 || mira == 0.4){
						OSMBCount++;
						errorByOSMB++;
					}else if(mira == 1.0) OSMACount++;
					else if (mira == 0.9) OSMBCount++;
					else throw new Error();
				}else throw new Error();
				totalCompared++;
			}
		}  
		if(totalCompared != totalSampleList.size()) throw new Error(totalCompared + " != " + totalSampleList.size());
		double totalByMira = 0.0;
		for(int i:totalSampleList) totalByMira += i;
		totalByMira /= totalCompared;
		System.out.println("=====================================");
		System.out.println("Submitted: " + filename + "\t" + name + "\t" + noiseSize);
		System.out.println("=====================================");
		System.out.println("Total Compared: " + totalCompared);
		System.out.println("OSMA Count: " + OSMACount);
		System.out.println("OSMB Count: " + OSMBCount);
		System.out.println("Avg Used by Mira: " + totalByMira);
		System.out.println("OSMA Error: " + errorByOSMA + "(" + Utils.roundToDecimals(errorByOSMA * 100.0 / totalCompared, 2) + "%)");
		System.out.println("OSMB Error: " + errorByOSMB + "(" + Utils.roundToDecimals(errorByOSMB * 100.0 / totalCompared, 2) + "%)");
		System.out.println("OSMA Error @ 50: " + errorAt50ByOSMA + "(" + Utils.roundToDecimals(errorAt50ByOSMA * 100.0 / totalCompared, 2) + "%)");
		System.out.println("OSMB Error @ 50: " + errorAt50ByOSMB + "(" + Utils.roundToDecimals(errorAt50ByOSMB * 100.0 / totalCompared, 2) + "%)");
		System.out.println("Total OSMA Error: " + (errorByOSMA + errorAt50ByOSMA) + "(" + 
				Utils.roundToDecimals((errorByOSMA + errorAt50ByOSMA) * 100.0 / totalCompared, 2) + "%)");
		System.out.println("Total OSMB Error: " + (errorByOSMB + errorAt50ByOSMB) + "(" + 
				Utils.roundToDecimals((errorByOSMB + errorAt50ByOSMB) * 100.0 / totalCompared, 2) + "%)");
		System.out.println("Total Error (without 50): " + (errorByOSMA + errorByOSMB) + "(" + 
				Utils.roundToDecimals((errorByOSMA + errorByOSMB) * 100.0 / totalCompared, 2) + "%)");
		System.out.println("Total Error (with 50): " + (errorByOSMA + errorByOSMB + errorAt50ByOSMA + errorAt50ByOSMB) + "(" + 
				Utils.roundToDecimals((errorByOSMA + errorByOSMB + errorAt50ByOSMA + errorAt50ByOSMB) * 100.0 / totalCompared, 2) + "%)");
		System.out.println("=====================================");
	}

	private List<Double> runCrossValidation(final int numOfFolds, Instances randData, int noiseSize, int baggingIterations, 
			List<Integer> totalSampleList) throws Exception{
		List<Double> name2PredictionList = new ArrayList<Double>();
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
			name2PredictionList.addAll(runPrediction(train, test, baggingIterations, totalSampleList));
		}
		return name2PredictionList;
	}

	private List<Double> runPrediction(Instances train, Instances test, int baggingIterations, List<Integer> totalSampleList) throws Exception{
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
			List<Classifier> cList = new ArrayList<Classifier>();
			for(int x = 0; x < baggingIterations; x++){
				if(x != 0 && x % 10000 == 0) System.out.print(x + " / " + baggingIterations + "...");
				Classifier c = Classifier.makeCopy(classifier);
				Instances tempInstances = getBootstrap(train, x);
				c.buildClassifier(tempInstances);
				cList.add(c);
			}
			/*
			 * Predicting
			 */
			System.out.print("Predicting...");
			List<Double> predictionList = new ArrayList<Double>();
			//				double totalSamples = 0;
			for(int x = 0; x < test.numInstances(); x++){
				if(x % 100 == 0) System.out.print(x + "...");
				int pos = 0;//for 1.0
				int neg = 0;//for 0.0
				for(int y = 0; y < cList.size(); y++){
					double predictionScore = cList.get(y).classifyInstance(test.instance(x));
					if(predictionScore == 1.0) pos++;
					else if(predictionScore == 0.0) neg++;
					else throw new Error();
				}
				if(pos > neg) predictionList.add(1.0);
				else if(pos < neg) predictionList.add(0.0);
				else predictionList.add(0.5);
			}
			System.out.println("Done!");
			return predictionList;
		}else{
			/*
			 * Run Mira
			 */
			return runMira(train, test, classifier, totalSampleList);
		}

	}

	private List<Double> runMira(Instances train, Instances test, Classifier baseClassifier, List<Integer> totalSampleList) throws Exception{
		List<Double> predictionList = new ArrayList<Double>();
		final double theta = 0.5;
		List<Classifier> trainedClassifierList = new ArrayList<Classifier>();
		System.out.print("Running Mira...");
		for(int x = 0; x < test.numInstances(); x++){
			Mira trainMira = new Mira(theta, alpha, beta, maxSamples);
			//Generate new train classifiers until a conclusion could be made
			int counter = 0;
			int pos = 0; //1.0
			int neg = 0; //0.0
			while(trainMira.obtainAnotherSample()){
				if(counter >= trainedClassifierList.size()){
					//Train Another Classifier
					Classifier cloneC = Classifier.makeCopy(baseClassifier);
					cloneC.buildClassifier(this.getBootstrap(train, counter));
					trainedClassifierList.add(cloneC);
				}
				double score = trainedClassifierList.get(counter).classifyInstance(test.instance(x));
				if(score == 1.0){
					pos++;
					trainMira.update(true, RuleStatus.APNOTSATISFIED);
				}else if(score == 0.0){
					neg++;
					trainMira.update(false, RuleStatus.APNOTSATISFIED);
				}else throw new Error();
				counter++;
			}
			//System.out.print(counter + ","); //Shows how many classifiers are used
			if(trainMira.concludeByOSMA()){
				if(pos > neg) predictionList.add(1.0);
				else if(pos < neg) predictionList.add(0.0);
				else predictionList.add(0.5);
			}else{
				if(pos > neg) predictionList.add(0.9);
				else if(pos < neg) predictionList.add(0.1);
				else predictionList.add(0.4);
			}
			totalSampleList.add(trainMira.getTotalSamples());
		}
		//System.out.println();
		System.out.print("Done!");
		return predictionList;
	}

	/*
	 * Do bootstrapping
	 */
	private Instances getBootstrap(Instances inst, int seed) throws Exception{
		Resample filter = new Resample();
		filter.setInputFormat(inst);
		filter.setRandomSeed(seed);
		return Filter.useFilter(inst, filter);
	}
}
