package section4;

import java.util.List;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;
import afc.utils.Arff;
import afc.utils.FileSystem;

public class Section4Run {
	/*
	 * Section 4: Show how Dynamic Bagging could achieve same results with a controllable error rate as say 1,000,000 number of classifiers
	 * 1) Increase the memory size
	 * 2) Make it such that it does not use so much memory if 1 could not solve the out of memory problem
	 */
	@Test
	public void run() throws Exception{
		/*
		 * Init for multi-threads
		 */
//		int nThreads = Runtime.getRuntime().availableProcessors();//Use all available processors
//		int nThreads = 1;
//		ExecutorService threadExecutor = Executors.newFixedThreadPool(nThreads);
//		List<Future<Results>> resultList = new ArrayList<Future<Results>>();
		/*
		 * 1) Read in arff file from data folder
		 * 2) Do 5 times ten-fold cross validation
		 * 3) Add noise of increasing size (i.e. 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%)
		 * 4) Compute the AUC
		 * PredictionStatsII(List<Integer> classList, List<Double> predictionScoreList)
		 */
		final int numOfResample = 1;
		final int numOfRepeat = 2;
		final int numOfFolds = 5; //Number of cross-validation
		final double errorRates = 0.0001;
		final String inputDir = "./data/";
		final int[] Iterations = {-1, 50000};
		final int[] noiseSize = {0};
		final int maxSamples = 1000;
		final String[] names = {"C45", "NN10", "NB"};
//		final String[] names = {"C45"};
		final Classifier[] classifiers = {new J48(), new IBk(10), new NaiveBayes()};
//		final Classifier[] classifiers = {new J48()};

		List<String> arffFileList = FileSystem.listDir(inputDir, true, true);
		for(String arffFile:arffFileList){
			if(arffFile.contains(".arff") == false) continue;
			String filename = FileSystem.getNameFromStringLocation(arffFile, false);
			System.out.println(filename);
			System.out.println("Reading " + arffFile + "...");
			Instances originalInstances = Arff.getAsInstances(arffFile, false);
			System.out.println("Sample Size: " + originalInstances.numInstances());
			/*
			 * Repeats
			 */
			for(int subsampleIndex = 0; subsampleIndex < numOfResample; subsampleIndex++){
				Instances instances;
				if(subsampleIndex == 0) instances = originalInstances;
				else{
					String[] options = new String[1];
					options[0] = "-S " + subsampleIndex;                                    // set the random seed
					Resample resample = new Resample();                         // new instance of filter
					resample.setOptions(options);                           // set options
					resample.setInputFormat(originalInstances);                          // inform filter about dataset **AFTER** setting options
					instances = Filter.useFilter(originalInstances, resample);   // apply filter
				}
				for(int nz:noiseSize){
					for(int i = 0; i < names.length; i++){
//						resultList.add(threadExecutor.submit(new Compute(nz, numOfRepeat, numOfFolds, instances, Iterations, filename, 
//								names[i], classifiers[i], maxSamples, errorRates)));
						new Compute(nz, numOfRepeat, numOfFolds, instances, Iterations, filename, 
								names[i], classifiers[i], maxSamples, errorRates);
					}
				}
			}
		}
//		threadExecutor.shutdown();
//		for(Future<Results> r:resultList){
//			r.get();
//		}
	}
}
