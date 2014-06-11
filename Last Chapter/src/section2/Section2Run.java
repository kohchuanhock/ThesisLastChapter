package section2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.Test;

import afc.basic.statistics.Mean;
import afc.basic.statistics.StandardDeviation;
import afc.graphing.r.R;
import afc.graphing.r.RPlotCI;

public class Section2Run {
	/*
	 * Section 2: Bagging generates bootstrap replicates that are enriched with “good” samples, a formal proof.
	 */
	@Test
	public void main(){
		int[] goodSamples = {50, 55, 60, 65, 70, 75, 80, 85, 90, 95}; //Number of good samples (in %) in the set
		final int[] baggingIterations = {10, 100, 1000}; //Iteration set for bagging
		final int[] totalSamples = {50, 100, 500, 1000}; //Total number of samples
//		int[] goodSamples = {75, 80, 85, 90, 95}; //Number of good samples (in %) in the set
		String outputDir = "./graphs/section2/overview/";
		run(goodSamples, outputDir, baggingIterations, totalSamples);
		goodSamples = new int[10];
		for(int i = 0; i < goodSamples.length; i++) goodSamples[i] = 80 + (i*2);
		outputDir = "./graphs/section2/detailed/";
		run(goodSamples, outputDir, baggingIterations, totalSamples);
	}
	
	@Test
	public void limsoonRequest(){
		int[] goodSamples = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99}; //Number of good samples (in %) in the set
//		int[] goodSamples = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; //Number of good samples (in %) in the set
		final int[] baggingIterations = {100, 200, 500, 1000, 2000, 5000}; //Iteration set for bagging
		final int[] totalSamples = {100}; //Total number of samples
		String outputDir = "./graphs/section2/special/";
		run(goodSamples, outputDir, baggingIterations, totalSamples);
	}
	
	@Test
	public void limsoonRequest2(){
		int[] goodSamples = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95}; //Number of good samples (in %) in the set
//		int[] goodSamples = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; //Number of good samples (in %) in the set
		final int[] baggingIterations = {1}; //Iteration set for bagging
		final int[] totalSamples = {100, 200, 500, 1000}; //Total number of samples
		String outputDir = "./graphs/section2/special2/";
		run(goodSamples, outputDir, baggingIterations, totalSamples);
	}

	public void run(final int[] goodSamples, final String outputDir, final int[] baggingIterations, final int[] totalSamples){
		/*
		 * Parameters
		 */
		final int repeats = 1000; //Number of repeats for each experiment settings - Need to have 1000 for real run
		final int foldValidation = 1; //Just be 1
//		final int[] totalSamples = {50, 100, 500, 1000}; //Total number of samples
		
		
		Map<String, List<Double>> settings2List = new HashMap<String, List<Double>>();
		int seed = 0;

		for(int ts:totalSamples){
			System.out.println("Total Samples: " + ts);
			for(int gs:goodSamples){
				System.out.println("Good Samples (%): " + gs);
				final int badSamples = (ts - (int)(gs / 100.0 * ts));
				List<Double> hTheoryList = new ArrayList<Double>();
				hTheoryList.add(computeH(badSamples, ts));
				List<Double> hPrimeTheoryList = new ArrayList<Double>();
				hPrimeTheoryList.add(computeHPrime(badSamples, ts));
				List<Double> hPrimePrimeTheoryList = new ArrayList<Double>();
				hPrimePrimeTheoryList.add(computeHPrimePrime(badSamples, ts));
				List<Double> hMinusHPrimeTheoryList = new ArrayList<Double>();
				hMinusHPrimeTheoryList.add(computeHMinusHPrime(badSamples, ts));
				
				for(int bi:baggingIterations){
					List<Double> hPercentList = new ArrayList<Double>();
					List<Double> hPrimePercentList = new ArrayList<Double>();
					List<Double> hPrimePrimePercentList = new ArrayList<Double>();
					List<Double> hBetterThanHPrimePercentList = new ArrayList<Double>();
					List<Double> hBetterThanOrEqualHPrimePercentList = new ArrayList<Double>();
					List<Double> hMinusHPrimePercentList = new ArrayList<Double>();
					for(int r = 0; r < repeats; r++){
						double hPercent = 0.0;
						double hPrimePercent = 0.0;
						double hPrimePrimePercent = 0.0;
						double hBetterThanHPrimePercent = 0.0;
						double hBetterThanOrEqualHPrimePercent = 0.0;
						double hMinusHPrimePercent = 0.0;
						for(int f = 0; f < foldValidation; f++){
							int[] results = efficientResampleAndComputeResults(gs, ts, bi, seed++);
							if(results[0] > results[2]) hBetterThanHPrimePercent++;
							if(results[0] >= results[2]) hBetterThanOrEqualHPrimePercent++;
							hPercent += (results[0] + 0.0) / bi;
							hPrimePrimePercent += (results[1] + 0.0) / bi;
							hPrimePercent += (results[2] + 0.0) / bi;
							hMinusHPrimePercent += (results[0] - results[2] + 0.0) / bi;
						}
						hPercent /= foldValidation;
						hPrimePrimePercent /= foldValidation;
						hPrimePercent /= foldValidation;
						hBetterThanHPrimePercent /= foldValidation;
						hBetterThanOrEqualHPrimePercent /= foldValidation;
						hMinusHPrimePercent /= foldValidation;

						hPercentList.add(hPercent);
						hPrimePrimePercentList.add(hPrimePrimePercent);
						hPrimePercentList.add(hPrimePercent);
						hMinusHPrimePercentList.add(hMinusHPrimePercent);
						hBetterThanHPrimePercentList.add(hBetterThanHPrimePercent);
						hBetterThanOrEqualHPrimePercentList.add(hBetterThanOrEqualHPrimePercent);
					}
					/*
					 * Store results
					 */
					settings2List.put(ts + "_" + gs + "_" + bi + "_H", hPercentList);
					settings2List.put(ts + "_" + gs + "_" + bi + "_HPrimePrime", hPrimePrimePercentList);
					settings2List.put(ts + "_" + gs + "_" + bi + "_HPrime", hPrimePercentList);
					settings2List.put(ts + "_" + gs + "_" + bi + "_HBetterThanHPrime", hBetterThanHPrimePercentList);
					settings2List.put(ts + "_" + gs + "_" + bi + "_HBetterThanOREqualHPrime", hBetterThanOrEqualHPrimePercentList);
					settings2List.put(ts + "_" + gs + "_HTheory", hTheoryList);
					settings2List.put(ts + "_" + gs + "_HPrimeTheory", hPrimeTheoryList);
					settings2List.put(ts + "_" + gs + "_HPrimePrimeTheory", hPrimePrimeTheoryList);
					settings2List.put(ts + "_" + gs + "_HMinusHPrimeTheory", hMinusHPrimeTheoryList);
					settings2List.put(ts + "_" + gs + "_" + bi + "_HMinusHPrime", hMinusHPrimePercentList);
				}
			}
		}
		/*
		 * Display results in graphs
		 */
		graph3(totalSamples, baggingIterations, goodSamples, settings2List, outputDir, "HMinusHPrime", "expression(\"P\"[B]*\"(<x) - P\"[B]*\"(>x)\")");
		/*
		 * 1) Individual graphs and also the display of obtained mean of h, h' and h''
		 */
		graph1(totalSamples, baggingIterations, goodSamples, settings2List, outputDir, repeats, "_HBetterThanHPrime", "h > h\\\'");
		graph1(totalSamples, baggingIterations, goodSamples, settings2List, outputDir, repeats, "_HBetterThanOREqualHPrime", "expression(h >= \"h\\'\")");
		graph1(totalSamples, baggingIterations, goodSamples, settings2List, outputDir, repeats, "_HMinusHPrime", "h - h\\\'");
		/*
		 * 2) Show the h, h' and h'' mean estimates
		 */
		graph2(totalSamples, baggingIterations, goodSamples, settings2List, outputDir, "H", "h");
		graph2(totalSamples, baggingIterations, goodSamples, settings2List, outputDir, "HPrime", "h\\\'");
		graph2(totalSamples, baggingIterations, goodSamples, settings2List, outputDir, "HPrimePrime", "h\\\'\\\'");
		/*
		 * 3)
		 */
		
	}

	/*
	 * Let p = y/m and q = x/m = (1−p)
	 * y is number of good samples and x is number of bad samples
	 * h = Summation of k>x [P(k)]
	 * h' = Summation of k<x [P(k)]
	 * h'' = P(x) = (mCx)(pm−x)(qx) = (m Choose x)(Math.pow(p, m-x)(Math.pow(q,x))) 
	 */
	public double computePk(int k, int numberOfBadSamples, int totalNumberOfSamples){
		/*
		 * computes P(x) = (mCx)(pm−x)(qx) = (m Choose x)(Math.pow(p, m-x)(Math.pow(q,x)))
		 */
		final int x = numberOfBadSamples;
		final int m = totalNumberOfSamples;
		final double q = (x + 0.0) / m;
		final double p = 1 - q;
		
//		if(k > m - k) return ArithmeticUtils.binomialCoefficient(m, m-k) * Math.pow(p, m - k) * Math.pow(q, k);
//		else return ArithmeticUtils.binomialCoefficient(m, k) * Math.pow(p, m - k) * Math.pow(q, k);
		return binomial(m, k, p, q);
	}

	public double binomial(int n, int k, double p, double q){
		double results = Math.pow(p, n - k) * Math.pow(q, k);
		if (k < 0 || k > n) return 0;
		if(k > n - k){
			k = n - k;
		}
		for(int i = 0; i < k; i++){
			results *= (n - (k - (i+1)));
			results /= i+1;
		}
		return results;
	}

	@Test
	public void test(){
//		System.out.println(binomial(100, 49));
//		System.out.println(ArithmeticUtils.binomialCoefficient(100, 49));
	}
	
	public double computeHMinusHPrime(int numberOfBadSamples, int totalNumberOfSamples){
		return computeH(numberOfBadSamples, totalNumberOfSamples) - computeHPrime(numberOfBadSamples, totalNumberOfSamples);
	}

	public double computeHPrime(int numberOfBadSamples, int totalNumberOfSamples){
		double total = 0.0;
		final int x = numberOfBadSamples;
		final int m = totalNumberOfSamples;
		for(int k = x + 1; k <= m; k++){
			total += computePk(k, x, totalNumberOfSamples);
		}
		return total;
	}

	public double computeH(int numberOfBadSamples, int totalNumberOfSamples){
		double total = 0.0;
		final int x = numberOfBadSamples;
		for(int k = 0; k < x; k++){
			total += computePk(k, x, totalNumberOfSamples);
		}
		return total;
	}

	public double computeHPrimePrime(int numberOfBadSamples, int totalNumberOfSamples){
		/*
		 * The number of times sampled bags contains the same number of good samples as the original bags
		 */
		return computePk(numberOfBadSamples, numberOfBadSamples, totalNumberOfSamples);
	}

	public List<Integer> populateData(int totalSamples, int goodSamplePercentage){
		int totalGoodSamples = (int)(goodSamplePercentage / 100.0 * totalSamples);
		//		System.out.println(totalGoodSamples + "\t" + goodSamplePercentage + "\t" + totalSamples);
		List<Integer> iList = new ArrayList<Integer>();
		for(int i = 0; i < totalSamples; i++){
			if(i < totalGoodSamples){
				iList.add(1);
			}else{
				iList.add(0);
			}
		}
		/*
		 * Check 1: summation of iList should be equal to the totalGoodSamples
		 */
		int sum = 0;
		for(int i:iList) sum += i;
		if(sum != totalGoodSamples) throw new Error("Error Check 1");
		/*
		 * Check 2: totalGoodSamples / 100.0 * totalSamples == goodSamplePercentage
		 */
		if(sum * 100.0 / totalSamples != goodSamplePercentage) throw new Error("Error Check 2: " + (sum * 100.0 / totalSamples) + "\t" + goodSamplePercentage);
		return iList;
	}

	public List<Integer> sampleWithReplacement(List<Integer> iList, int baggingIterations, int seed){
		Random rand = new Random(seed);
		List<Integer> goodSampleSizeList = new ArrayList<Integer>();
		/*
		 * Repeat the process for the given number of iterations
		 */
		for(int i = 0; i < baggingIterations; i++){
			/*
			 * Re-sample s set of same size
			 */
			int numberOfGoodSamples = 0;
			for(int j = 0; j < iList.size(); j++){
				int index = rand.nextInt(iList.size());
				numberOfGoodSamples += iList.get(index);
			}
			/*
			 * It should not be the case where the number of good samples is > iList.size
			 */
			if(numberOfGoodSamples > iList.size()) throw new Error("Error Check 3");
			goodSampleSizeList.add(numberOfGoodSamples);
		}
		return goodSampleSizeList;
	}

	public int[] computeResults(List<Integer> goodSampleSizeList, int goodSamplePercentage, int totalSamples){
		int goodSamples = (int)(goodSamplePercentage / 100.0 * totalSamples);
		int betterCount = 0;
		int equalCount = 0;
		int worseCount = 0;
		for(int i:goodSampleSizeList){
			if(i > goodSamples){
				betterCount++;
			}else if(i == goodSamples){
				equalCount++;
			}else{
				worseCount++;
			}
		}
		return new int[]{betterCount, equalCount, worseCount};
	}

	public int[] efficientResampleAndComputeResults(int goodSamplePercentage, int totalSamples, int baggingIterations, int seed){
		int goodSamples = (int)(goodSamplePercentage / 100.0 * totalSamples);
		int betterCount = 0;
		int equalCount = 0;
		int worseCount = 0;
		Random rand = new Random(seed); // So that the outcome will be reproducible
		for(int i = 0; i < baggingIterations; i++){
			/*
			 *	Re-sampling with replacements  
			 */
			int total = 0;
			for(int j = 0; j < totalSamples; j++){
				if(rand.nextInt(totalSamples) < goodSamples) total++;
			}
			if(total > goodSamples) betterCount++;
			else if(total == goodSamples) equalCount++;
			else worseCount++;
		}
		return new int[]{betterCount, equalCount, worseCount};
	}

	public void displayResultsInTextForm(List<Integer> goodSampleSizeList, int goodSamples, int baggingIterations, int[] results, int totalSamples){
		double mean = Mean.compute(goodSampleSizeList);
		double sd = StandardDeviation.compute(goodSampleSizeList);
		System.out.println();
		System.out.println("Real: " + (goodSamples / 100.0 * totalSamples));
		System.out.println("Mean: " + mean);
		System.out.println("SD: " + sd);
		System.out.println("Better Count: " + results[0] + " / " + baggingIterations);
		System.out.println("Equal Count: " + results[1] + " / " + baggingIterations);
		System.out.println("Worse Count: " + results[2] + " / " + baggingIterations);
		System.out.println();
	}

	public void graph1(int[] totalSamples, int[] baggingIterations, int[] goodSamples, Map<String, List<Double>> settings2ResultsList, String outputDir, 
			int repeats, String suffix, String yLabel){
		R r = new R();
		for(int ts:totalSamples){
			List<Double> xList = new ArrayList<Double>();
			List<Double> yList = new ArrayList<Double>();
			List<String> gList = new ArrayList<String>();
			for(int bi:baggingIterations){
				for(int gs:goodSamples){
					List<Double> rList = settings2ResultsList.get(ts + "_" + gs + "_" + bi + suffix);
					/*
					 * Temp
					 */
//					double t = 0.0;
//					for(double d:rList){
//						if(d > 0.001 && d < 1.0) throw new Error("Neither 0.0 nor 1.0 but " + d);
//						t += d;
//					}
//					xList.add(gs + 0.0);
//					yList.add(t);
//					gList.add(bi + "");
					for(double d:rList){
						xList.add(gs + 0.0);
						yList.add(d);
						gList.add(bi + "");
					}
				}
			}
			StringBuffer sb = RPlotCI.plotCI2(outputDir + "TotalSamples" + ts + suffix + ".pdf", xList, yList, gList, "No. of Good samples (in %, i.e., p)", 
					yLabel, "Bootstrap_replicates", "Total Samples = " + ts + ", Repeats = " + repeats, 0.5);
			r.runCode(sb, true);
		}
	}

	public void graph2(int[] totalSamples, int[] baggingIterations, int[] goodSamples, Map<String, List<Double>> settings2ResultsList, String outputDir,
			String type, String typeLabel){
		R r = new R();
		for(int ts:totalSamples){
			List<Double> xList = new ArrayList<Double>();
			List<Double> yList = new ArrayList<Double>();
			List<String> gList = new ArrayList<String>();
			for(int bi:baggingIterations){
				for(int gs:goodSamples){
					List<Double> rList = settings2ResultsList.get(ts + "_" + gs + "_" + bi + "_" + type);
					for(double d:rList){
						xList.add(gs + 0.0);
						yList.add(d);
						gList.add(bi + "");
					}
				}
			}
			for(int gs:goodSamples){
				List<Double> rList = settings2ResultsList.get(ts + "_" + gs + "_" + type + "Theory");
				for(double d:rList){
					xList.add(gs + 0.0);
					yList.add(d);
					gList.add("Theory");
				}
			}
			StringBuffer sb = RPlotCI.plotCI2(outputDir + "TotalSamples" + ts + "_" + type + ".pdf", xList, yList, gList, "No. of Good samples (in %, i.e., p)", 
					typeLabel, "Bootstrap_replicates", "Total Samples = " + ts, 0.5);
			r.runCode(sb, true);
		}
	}
	
	public void graph3(int[] totalSamples, int[] baggingIterations, int[] goodSamples, Map<String, List<Double>> settings2ResultsList, String outputDir,
			String type, String typeLabel){
		R r = new R();
		List<Double> xList = new ArrayList<Double>();
		List<Double> yList = new ArrayList<Double>();
		List<String> gList = new ArrayList<String>();
		List<String> graphNameOrderList = new ArrayList<String>();
		for(int ts:totalSamples){
			graphNameOrderList.add(ts + "");
			System.out.println(ts);
			for(int gs:goodSamples){
				List<Double> rList = settings2ResultsList.get(ts + "_" + gs + "_" + type + "Theory");
				for(double d:rList){
					xList.add(gs + 0.0);
					yList.add(d);
					gList.add(ts + "");
				}
			}
		}
		StringBuffer sb = RPlotCI.plotCI2(outputDir + "Theory_" + type + ".pdf", xList, yList, gList, "No. of Good samples (in %, i.e., p)", 
				typeLabel, "m", "Theory", 0.5, graphNameOrderList);
		r.runCode(sb, true);
	}
}
