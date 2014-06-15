package section1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Test;

import section3.Results;
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
import afc.utils.Arff.AFTERUSE;

public class Section1Run {
	/*
	 * Section 1: Both stable and unstable algorithms are well behaved.
	 * 
	 * Demonstrate that as noise increases, algorithms accuracy reduces for both stable (kNN) and unstable algorithms (C4.5)
	 */
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
		 * 2) Do 10 times ten-fold cross validation
		 * 3) Add noise of increasing size (i.e. 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%)
		 * 4) Compute the AUC
		 * PredictionStatsII(List<Integer> classList, List<Double> predictionScoreList)
		 */
		final double positionShift = 1.0;
		final int numOfResample = 1;
		final int numOfFolds = 5;
		final int numOfRepeat = 5;
		final int noiseSizeStart = 0;
		final int noiseSizeEnd = 100;
		final int noiseSizeInterval = 10;
		final String outputDir = "./graphs/section1/";
		final String inputDir = "./data/";
		final String[] names = {"C45", "ANN", "NN10", "NB"};
//		final String[] names = {"C45", "NN10"};
		final Classifier[] classifiers = {new J48(), new MultilayerPerceptron(), new IBk(10), new NaiveBayes()};
//		final Classifier[] classifiers = {new J48(), new IBk(10)};
		
		List<String> arffFileList = FileSystem.listDir(inputDir, true, true);
		Map<String, List<Double>> string2AUCList = new HashMap<String, List<Double>>();
		Map<String, List<Double>> string2MeanAUCList = new HashMap<String, List<Double>>();
		int fileSize = 0;
		for(String arffFile:arffFileList){
			/*
			 * For each arff file found in Directory
			 */
			if(arffFile.contains(".arff") == false) continue;
			fileSize++;
			String filename = FileSystem.getNameFromStringLocation(arffFile, false);
			System.out.println(filename);
			System.out.println("Reading " + arffFile + "...");
			Instances originalInstances = Arff.getAsInstances(arffFile, AFTERUSE.DO_NOTHING);
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
				for(int noiseSize = noiseSizeStart; noiseSize <= noiseSizeEnd; noiseSize += noiseSizeInterval){
					System.out.println("Noise Size: " + noiseSize + "%");
					resultList.add(threadExecutor.submit(new Compute(numOfRepeat, numOfFolds, instances, names, classifiers, noiseSize, 
							filename, subsampleIndex, string2AUCList, string2MeanAUCList)));
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
		/*
		 * 1) Individual
		 * 		Mean of each file VS Noise Size
		 */
//		R r = new R();
//		for(String arffFile:arffFileList){
//			String filename = Utils.getNameFromStringLocation(arffFile, false);
//			if(arffFile.contains(".arff") == false) continue;
//			List<Double> xAxisList = new ArrayList<Double>();
//			List<Double> yAxisList = new ArrayList<Double>();
//			List<String> graphNameList = new ArrayList<String>();
//			for(int subsampleIndex = 0; subsampleIndex < numOfResample; subsampleIndex++){
//				for(int noiseSize = noiseSizeStart; noiseSize <= noiseSizeEnd; noiseSize += noiseSizeInterval){
//					for(int i = 0; i < names.length; i++){
//						List<Double> aucList = string2AUCList.get(filename + subsampleIndex + noiseSize + names[i]);
//						if(aucList == null){
//							System.out.println("SearchKey: " + filename + subsampleIndex + noiseSize + names[i]);
//							for(String n:string2AUCList.keySet()) System.out.println(n);
//						}
//						for(double d:aucList){
//							xAxisList.add(noiseSize + 0.0);
//							yAxisList.add(d);;
//							graphNameList.add(names[i]);
//						}	
//					}
//				}
//			}
//			StringBuffer sb = new StringBuffer();
//			sb = RPlotCI.plotCI2(outputDir + filename + ".pdf", xAxisList, yAxisList, graphNameList, "Noise Size", "AUC", "Classifiers", 
//					numOfRepeat + " repeats of " + numOfFolds + " x-validation", positionShift);
//			r.runCode(sb);
//		}
		/*
		 * 3) Overall
		 * 		Mean of all the various files VS Noise Size
		 */
//		StringBuffer sb = new StringBuffer();
//		List<Double> xAxisList = new ArrayList<Double>();
//		List<Double> yAxisList = new ArrayList<Double>();
//		List<Double> yAxisChangeList = new ArrayList<Double>();
//		List<String> graphNameList = new ArrayList<String>();
//		for(int noiseSize = noiseSizeStart; noiseSize <= noiseSizeEnd; noiseSize += noiseSizeInterval){
//			for(int i = 0; i < names.length; i++){
//				List<Double> aucList = string2MeanAUCList.get(noiseSize + names[i]);
//				List<Double> baseAUCList = string2MeanAUCList.get(0 + names[i]);
//				for(int j = 0; j < aucList.size(); j++){
//					graphNameList.add(names[i]);
//					yAxisList.add(aucList.get(j));
//					yAxisChangeList.add(aucList.get(j) - baseAUCList.get(j));
//					xAxisList.add(noiseSize + 0.0);
//				}
//			}
//		}
		
		R r = new R();
		List<Double> xAxisList = new ArrayList<Double>();
		List<Double> yAxisList = new ArrayList<Double>();
		List<Double> yAxisChangeList = new ArrayList<Double>();
		List<String> graphNameList = new ArrayList<String>();
		for(String arffFile:arffFileList){
			String filename = FileSystem.getNameFromStringLocation(arffFile, false);
			if(arffFile.contains(".arff") == false) continue;
			for(int subsampleIndex = 0; subsampleIndex < numOfResample; subsampleIndex++){
				for(int noiseSize = noiseSizeStart; noiseSize <= noiseSizeEnd; noiseSize += noiseSizeInterval){
					for(int i = 0; i < names.length; i++){
						List<Double> aucList = string2AUCList.get(filename + subsampleIndex + noiseSize + names[i]);
						if(aucList == null){
							System.out.println("SearchKey: " + filename + subsampleIndex + noiseSize + names[i]);
							for(String n:string2AUCList.keySet()) System.out.println(n);
						}
						List<Double> baseAUCList = string2AUCList.get(filename + subsampleIndex + 0 + names[i]);
						for(int j = 0; j < aucList.size(); j++){
							graphNameList.add(names[i]);
							yAxisList.add(aucList.get(j));
							yAxisChangeList.add(aucList.get(j) - baseAUCList.get(j));
							xAxisList.add(noiseSize + 0.0);
						}
					}
				}
			}
		}
		StringBuffer sb = RPlotCI.plotCI2(outputDir + "Overall.pdf", xAxisList, yAxisList, graphNameList, "Noise Size", "AUC", "Classifiers", 
				numOfRepeat + " repeats of " + numOfFolds + " x-validation", positionShift);
		r.runCode(sb);
		
		sb = RPlotCI.plotCI2(outputDir + "OverallChange.pdf", xAxisList, yAxisChangeList, graphNameList, "Noise Size", "AUC Change", "Classifiers", 
				numOfRepeat + " repeats of " + numOfFolds + " x-validation", positionShift);
		r.runCode(sb);
		
		int tCount = 0;
		for(int i = 0; i < graphNameList.size(); i++){
			if(graphNameList.get(i).equals(names[0]) && xAxisList.get(i) == 0.0) tCount++;
		}
		int expected = numOfRepeat * fileSize;
		if(tCount != expected) throw new Error(tCount + " != " + expected);
		System.out.println("===================================");
		System.out.println("No. Data Per Point: " + tCount);
		System.out.println("===================================");
	}
}
