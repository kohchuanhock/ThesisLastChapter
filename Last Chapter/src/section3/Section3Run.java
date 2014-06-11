package section3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;
import afc.graphing.r.R;
import afc.graphing.r.RPlotCI;
import afc.utils.Arff;
import afc.utils.FileSystem;

public class Section3Run {
	/*
	 * Section 3: Show how Dynamic Bagging could achieve similar results with much lesser iterations 
	 * and without the need to give the arbitrary fixed number of iterations
	 */
	@Test
	public void run() throws Exception{
		/*
		 * Init for multi-threads
		 */
		int nThreads = Runtime.getRuntime().availableProcessors();//Use all available processors
		//		int nThreads = 1;
		ExecutorService threadExecutor = Executors.newFixedThreadPool(nThreads);
		List<Future<Results>> resultList = new ArrayList<Future<Results>>();
		/*
		 * 1) Read in arff file from data folder
		 * 2) Do 10 times ten-fold cross validation
		 * 3) Add noise of increasing size (i.e. 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%)
		 * 4) Compute the AUC
		 * PredictionStatsII(List<Integer> classList, List<Double> predictionScoreList)
		 */
		final int numOfResample = 1;
		final int numOfRepeat = 2;
		final int numOfFolds = 5;
		final String outputDir = "./graphs/section3/";
		final String inputDir = "./data/";
		final int[] Iterations = {1, 10, 100, -1};
		final String[] iterationsName = {"1", "10", "100", "Dynamic"};
//		final int[] Iterations = {1};
//		final String[] iterationsName = {"1"};
		final int[] noiseSize = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
//				final int[] noiseSize = {0, 10};
		final int maxSamples = 100;
		final String[] names = {"C45", "ANN", "NN10", "NB"};
//		final String[] names = {"C45", "NN10"};
		final Classifier[] classifiers = {new J48(), new MultilayerPerceptron(), new IBk(10), new NaiveBayes()};
//		final Classifier[] classifiers = {new J48(), new IBk(10)};

		List<String> arffFileList = FileSystem.listDir(inputDir, true, true);
		Map<String, List<Double>> string2AUCList = new HashMap<String, List<Double>>();
		Map<String, List<Double>> string2ClassifierUsedList = new HashMap<String, List<Double>>();
		for(int bi:Iterations){
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
						System.out.println("=====================================");
						System.out.println("Submitted: " + filename + "\t" + bi + "\t" + nz);
						System.out.println("=====================================");
						resultList.add(threadExecutor.submit(new Compute(nz, numOfRepeat, numOfFolds, instances, bi, filename, subsampleIndex, 
								string2AUCList, string2ClassifierUsedList, names, classifiers, maxSamples)));
					}
				}
			}
		}
		threadExecutor.shutdown();
		for(Future<Results> r:resultList){
			r.get();
		}
		/*
		 * Plot Graphs
		 */
		plotGraph1(arffFileList, Iterations, numOfResample, string2AUCList, outputDir, noiseSize, numOfRepeat, 
				string2ClassifierUsedList, numOfFolds, names, iterationsName);
		//		plotGraph2(Iterations, string2MeanAUCList, outputDir, numOfRepeat, noiseSize);
		//		plotGraph3(Iterations, string2MeanAUCList, outputDir, numOfRepeat, noiseSize);
	}

	private void plotGraph1(List<String> arffFileList, int[] Iterations, int numOfResample, Map<String, List<Double>> string2AUCList,
			String outputDir, int[] noiseSize, int repeats, Map<String, List<Double>> string2ClassifierUsedList, int numOfFolds, String[] names,
			String[] iterationsNames){
		/*
		 * Individual
		 * 		Mean of each file VS Noise Size
		 * 
		 * 1) *_C45 and *_NN5
		 * 2) *_Overall 
		 */
		R r = new R();
		
		/*
		 * Create OverallByIterations.pdf
		 */
		List<Double> xAxisList = new ArrayList<Double>();
		List<Double> yAxisList = new ArrayList<Double>();
		List<String> graphNameList = new ArrayList<String>();

		List<Double> sampleXList = new ArrayList<Double>();
		List<Double> sampleYList = new ArrayList<Double>();
		List<String> sampleGraphNameList = new ArrayList<String>();
		int arffCount = 0;
		for(String arffFile:arffFileList){
			if(arffFile.contains(".arff") == false) continue;
			arffCount++;
			String filename = FileSystem.getNameFromStringLocation(arffFile, false);
			for(int a = 0; a < Iterations.length; a++){
				int bi = Iterations[a];
				//			for(int bi:Iterations){
				for(int subsampleIndex = 0; subsampleIndex < numOfResample; subsampleIndex++){
					for(int nz:noiseSize){
						for(int i = 0; i < names.length; i++){
							List<Double> aucList = string2AUCList.get(filename + "_" + subsampleIndex + "_" + nz + "_" + bi + "_" + names[i]);
							if(aucList.size() != repeats) throw new Error(aucList.size() + " != " + repeats);
							for(double d:aucList){
								xAxisList.add(nz + 0.0);
								yAxisList.add(d);
								graphNameList.add(iterationsNames[a]);
							}

							List<Double>  sList = string2ClassifierUsedList.get(filename + "_" + subsampleIndex + "_" + nz + "_" + bi + "_" + names[i]);
							for(double s:sList){
								sampleXList.add(nz + 0.0);
								sampleYList.add(s + 0.0);
								sampleGraphNameList.add(iterationsNames[a]);
								//								System.out.print(s + ",");
							}
							System.out.println();
						}
					}
				}
			}
		}
		/*
		 * Checking
		 */
		int tCount = 0;
		for(int i = 0; i < graphNameList.size(); i++){
			if(graphNameList.get(i).equals("1") && xAxisList.get(i) == 0.0) tCount++;
		}
		int expected = arffCount * numOfResample * names.length * repeats;
		if(tCount != expected) throw new Error(tCount + " != " + expected);
		StringBuffer sb = RPlotCI.plotCI2(outputDir + "OverallByIterations.pdf", xAxisList, yAxisList, graphNameList, "Noise Size", "AUC", "Iterations", 
				//				repeats + " repeats of " + numOfFolds + " x-validation",
//				numOfFolds + " x-validation",
				"",
				0.5);
		r.runCode(sb, true);

		sb = RPlotCI.plotCI2(outputDir + "OverallByIterationsSamples.pdf", sampleXList, sampleYList, sampleGraphNameList, "Noise Size", 
				"Average Iterations", "Iterations", 
				//				repeats + " repeats of " + numOfFolds + " x-validation",
//				numOfFolds + " x-validation",
				"",
				0.5);
		r.runCode(sb, true);

		/*
		 * Create OverallByArffFile.pdf
		 */
		for(String arffFile:arffFileList){
			if(arffFile.contains(".arff") == false) continue;
			xAxisList = new ArrayList<Double>();
			yAxisList = new ArrayList<Double>();
			List<Double> yChangeList = new ArrayList<Double>();
			graphNameList = new ArrayList<String>();
			
			sampleXList = new ArrayList<Double>();
			sampleYList = new ArrayList<Double>();
			sampleGraphNameList = new ArrayList<String>();
			String filename = FileSystem.getNameFromStringLocation(arffFile, false);
			for(int a = 0; a < Iterations.length; a++){
				int bi = Iterations[a];
				for(int subsampleIndex = 0; subsampleIndex < numOfResample; subsampleIndex++){
					for(int nz:noiseSize){
						for(int i = 0; i < names.length; i++){
							List<Double> aucList = string2AUCList.get(filename + "_" + subsampleIndex + "_" + nz + "_" + bi + "_" + names[i]);
							if(aucList.size() != repeats) throw new Error(aucList.size() + " != " + repeats);
							List<Double> baseAUCList = string2AUCList.get(filename + "_" + subsampleIndex + "_" + 0 + "_" + bi + "_" + names[i]);
							for(int j = 0; j < aucList.size(); j++){
								xAxisList.add(nz + 0.0);
								yAxisList.add(aucList.get(j));
								yChangeList.add(aucList.get(j) - baseAUCList.get(j));
								graphNameList.add(iterationsNames[a]);
							}
							List<Double>  sList = string2ClassifierUsedList.get(filename + "_" + subsampleIndex + "_" + nz + "_" + bi + "_" + names[i]);
							for(double s:sList){
								sampleXList.add(nz + 0.0);
								sampleYList.add(s + 0.0);
								sampleGraphNameList.add(iterationsNames[a]);
								//								System.out.print(s + ",");
							}
						}
					}
				}
			}
			sb = RPlotCI.plotCI2(outputDir + filename + "_Overall.pdf", xAxisList, yAxisList, graphNameList, "Noise Size", "AUC", "Iterations", 
					//				repeats + " repeats of " + numOfFolds + " x-validation"
//					numOfFolds + " x-validation",
					"",
					0.5);
			r.runCode(sb, true);

			sb = RPlotCI.plotCI2(outputDir + filename + "_OverallChange.pdf", xAxisList, yChangeList, graphNameList, "Noise Size", "AUC Change", "Iterations", 
//					repeats + " repeats of " + numOfFolds + " x-validation",
//					numOfFolds + " x-validation",
					"",
					0.5);
			r.runCode(sb, true);
			
			sb = RPlotCI.plotCI2(outputDir + filename + "_OverallSamples.pdf", sampleXList, sampleYList, sampleGraphNameList, "Noise Size", 
					"Average Iterations", "Iterations", 
					//				repeats + " repeats of " + numOfFolds + " x-validation",
//					numOfFolds + " x-validation",
					"",
					0.5);
			r.runCode(sb, true);
		}
		
		System.out.println("===================================");
		System.out.println("No. Data Per Point: " + tCount);
		System.out.println("===================================");
	}
}
