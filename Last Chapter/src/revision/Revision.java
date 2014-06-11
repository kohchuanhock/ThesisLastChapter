package revision;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Test;

import section3.Results;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;
import afc.utils.Arff;
import afc.utils.FileSystem;

public class Revision {
	@Test
	public void run() throws Exception{
		/*
		 * Init for multi-threads
		 */
		int nThreads = Runtime.getRuntime().availableProcessors();//Use all available processors
		ExecutorService threadExecutor = Executors.newFixedThreadPool(nThreads);
		List<Future<Results>> resultList = new ArrayList<Future<Results>>();
		/*
		 * 1) Read in arff file from data folder
		 * 2) Do 3 times three-fold cross validation
		 * 3) Compute the AUC of full training data
		 * 4) Compute the change in prediction for each instance being left out
		 * 5) Compute the AUC with all negative change being left out
		 */
		final int numOfResample = 1;
		final int numOfFolds = 3;
		final int numOfRepeat = 1;
		final String inputDir = "./data/";
		final String[] names = {"C45", "NN10", "NB"};
		final Classifier[] classifiers = {new J48(), new IBk(10), new NaiveBayes()};
		
		List<String> arffFileList = FileSystem.listDir(inputDir, true, true);
		for(String arffFile:arffFileList){
			/*
			 * For each arff file found in Directory
			 */
			if(arffFile.contains(".arff") == false) continue;
			String filename = FileSystem.getNameFromStringLocation(arffFile, false);
			System.out.println(filename);
			System.out.println("Reading " + arffFile + "...");
			Instances originalInstances = Arff.getAsInstances(arffFile, false);
			System.out.println("Sample Size: " + originalInstances.numInstances());
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
				resultList.add(threadExecutor.submit(new Compute(numOfRepeat, numOfFolds, instances, names, classifiers, 
						filename)));
			}
		}
		threadExecutor.shutdown();
		for(Future<Results> r:resultList){
			r.get();
		}
	}
}
