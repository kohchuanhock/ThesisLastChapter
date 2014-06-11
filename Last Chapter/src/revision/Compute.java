package revision;

/*
 * The classifier learning algorithms have their respective biases and, even when given noise-free data, may not learn the correct decision model. 
 * That is, due to the imperfection of classifier learning algorithms, 
 * so noise-free samples can be harmful to them and cause them to learn incorrect decision models. 
 * The samples that are harmful to different learning algorithms may be different, 
 * as different learning algorithms have different technical imperfections.  
 * 
 * So the theoretical proof that we have put forwarded can be applied to harmful samples verbatim. 
 * I.e., each bootstrap bag is likely to have fewer harmful samples than the original training data, 
 * so long as the number of harmful samples is relatively few in the original training data.
 * 
 * The expt validation is more difficult. 
 * In the noise case, we can inject noise by switching class labels. 
 * But here, we need to identify the harmful samples wrt each learning algorithms separately; 
 * then repeat our noise expts for this learning algorithm by treating these harmful samples like noise samples.
 * 
 * One way to proceed may be� Given a training data and a learning algorithm. 
 * Remove sample x. Apply the learning algorithm on the rest of training data. 
 * Test on test data. Record the change in accuracy. Rank all the samples based on the change in accuracy.
 * The biggest negative change ones are the harmful ones.
 *   
 * We then show the impact on the classifier as we remove n such harmful samples. 
 * We also show the bootstrap bags contain less of these harmful samples. 
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import section3.Results;
import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveRange;
import afc.predictionstatistics.PredictionStats;

public class Compute implements Callable<Results>{
	private int numOfRepeat;
	private int numOfFolds;
	private Instances instances;
	private String[] names;
	private Classifier[] classifiers;
	private String filename;

	public Compute(final int numOfRepeat, final int numOfFolds, Instances instances, String[] names, Classifier[] classifiers, String filename){
		this.numOfRepeat = numOfRepeat;
		this.numOfFolds = numOfFolds;
		this.instances = instances;
		this.names = names;
		this.classifiers = classifiers;
		this.filename = filename;
		if(this.filename.length() < 8) this.filename += "\t";
	}

	@Override
	public Results call() throws Exception {
		runNoise();
		return null;
	}

	private Map<String, List<Double>> runNoise() throws Exception{
		Map<String, List<Double>> name2AUCList = new HashMap<String, List<Double>>();
		for(int r = 0; r < numOfRepeat; r++){
			Random rand = new Random(r);   // create seeded number generator
			Instances randData = new Instances(instances);   // create copy of original data
			randData.randomize(rand);
			randData.stratify(numOfFolds);
			runCrossValidation(numOfFolds, randData, names, classifiers);
		}  
		return name2AUCList;
	}

	private Map<String, List<Double>> runCrossValidation(final int numOfFolds, Instances randData, String[] names, 
			Classifier[] classifiers) throws Exception{
		Map<String, List<Double>> name2PredictionList = new HashMap<String, List<Double>>();
		/*
		 * For each classifier type
		 */
		for(int i = 0; i < names.length; i++){
			/*
			 * For each fold (1/3 each)
			 */
			for (int n = 0; n < numOfFolds; n++) {		
				Instances train = randData.trainCV(numOfFolds, n);
				Instances test = randData.testCV(numOfFolds, n);
				List<Integer> trueClassList = new ArrayList<Integer>();
				for(int a = 0; a < test.numInstances(); a++) trueClassList.add((int)test.instance(a).classValue());
				PredictionStats stats = new PredictionStats(trueClassList, runPrediction(train, test, Classifier.makeCopy(classifiers[i])));
				double fullAUC = stats.computeAUC();
				
				Instances subRandData = new Instances(train);   // create copy of original data
				int negAUCSamples = 0;
				List<ScoreAndIndex> negAUCIndexList = new ArrayList<ScoreAndIndex>();
				for(int x = 0; x < train.numInstances(); x++){
					Instances subTrain = subRandData.trainCV(train.numInstances(), x);
					stats = new PredictionStats(trueClassList, runPrediction(subTrain, test, Classifier.makeCopy(classifiers[i])));
					double subAUC = stats.computeAUC();
					if(fullAUC * 1.00 < subAUC){
						negAUCSamples++;
						negAUCIndexList.add(new ScoreAndIndex(x, subAUC - fullAUC));
					}
				}
				double afterAUC = fullAUC;
				if(negAUCIndexList.size() > 0){
					String debug;
					String[] options = new String[2];
					options[0] = "-R";                                    // "range"
					options[1] = "";                                     // first attribute
					Collections.sort(negAUCIndexList, new SortByScore());
					int top = 3;
					for(int x = 0; x < negAUCIndexList.size() && (top == -1 || x < top); x++){
						if(x != 0) options[1] += ",";
						//+1 because index starts from 1
						options[1] += (negAUCIndexList.get(x).getIndex() + 1);
					}
					debug = options[1].substring(0);
					RemoveRange remove = new RemoveRange();                         // new instance of filter
					remove.setOptions(options);                           // set options
					remove.setInputFormat(train);                          // inform filter about dataset **AFTER** setting options
					Instances newData = null;
					try{
						newData = Filter.useFilter(train, remove);   // apply filter
					}catch(Exception e){throw new Error(train.numInstances() + "\t" + debug + "END");}
					System.out.println(newData.numInstances());
					
					stats = new PredictionStats(trueClassList, runPrediction(newData, test, Classifier.makeCopy(classifiers[i])));
					afterAUC = stats.computeAUC();
					double percentage = Utils.roundDouble((negAUCSamples * 100.0) / train.numInstances(), 2);
					System.out.println(this.filename + "\t" + names[i] + "\t" + n + "\t" + percentage + 
							"% (" + negAUCSamples + "/" + train.numInstances() + ")" + "\t" + Utils.roundDouble(fullAUC, 4) + " => " + 
							Utils.roundDouble(afterAUC, 4));
				}
			}
		}
		return name2PredictionList;
	}

	/*
	 * Get the prediction for each instances and gather them
	 */
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

class ScoreAndIndex{
	private double score;
	private int index;
	public ScoreAndIndex(int index, double score){
		this.score = score;
		this.index = index;
	}
	public int getIndex(){return index;}
	public double getScore(){return score;}
	public String toString(){return this.index + "\t" + this.score;}
}

class SortByScore implements Comparator<ScoreAndIndex>{
	@Override
	public int compare(ScoreAndIndex f1, ScoreAndIndex f2){
		if(f1.getScore() == f2.getScore()) return 0;
		else if(f1.getScore() < f2.getScore()) return 1;
		else return -1;
	}
}