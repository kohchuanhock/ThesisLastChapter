package section2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

public class Section2RunB {

	@Test
	public void testRun(){
		final int repeats = 1000;
		final int iterationSize[] = {100, 200, 300, 400, 500, 1000};
		final int sampleSize = 100;
		final int goodSampleSize[] = {50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100};
		/*
		 * Print Headers
		 */
		System.out.print("\tGood Sample Size:\t");
		for(int i:goodSampleSize) System.out.print(i + "\t");
		System.out.println();
		System.out.println();
		for(int numberOfBags:iterationSize){
			List<Integer> greaterList = new ArrayList<Integer>();
			List<Integer> greaterOrEqualList = new ArrayList<Integer>();
			for(int currentGoodSampleSize:goodSampleSize){
				int hGreaterThanHPrime = 0;
				int hGreaterThanOrEqualHPrime = 0;
				for(int r = 0; r < repeats; r++){
					int h = 0;
					int hPrime = 0;
					int hPrimePrime = 0;
					Random rand = new Random(r);
					for(int i = 0; i < numberOfBags; i++){
						/*
						 * sample with replacement
						 */
						int totalGoodSamples = 0;
						for(int s = 0; s < sampleSize; s++){
							if(rand.nextInt(sampleSize) < currentGoodSampleSize){
								/*
								 * Suppose that first n samples are good.
								 */
								totalGoodSamples++;
							}
						}
						//This bag has more "good" samples than original training data
						if(totalGoodSamples > currentGoodSampleSize) h++; 
						//This bag has same number of "good" samples as original training data
						else if(totalGoodSamples == currentGoodSampleSize) hPrimePrime++;
						//This bag has lesser "good" samples than original training data
						else hPrime++; 
					}
					if(h > hPrime) hGreaterThanHPrime++;//This run has more bags with more "good" samples than bags with less "good" samples
					if(h >= hPrime) hGreaterThanOrEqualHPrime++;
					System.out.println(hGreaterThanHPrime + "\t" + hPrimePrime + "\t" + hGreaterThanOrEqualHPrime);
				}
				greaterList.add(hGreaterThanHPrime);
				greaterOrEqualList.add(hGreaterThanOrEqualHPrime);
			}
			/*
			 * Display results
			 */
			System.out.print("Iterations: " + numberOfBags + " Greater:\t");
			for(int i:greaterList) System.out.print(i + "\t");
			System.out.println();
			System.out.print("Iterations: " + numberOfBags + "   Equal:\t");
			for(int i:greaterOrEqualList) System.out.print(i + "\t");
			System.out.println();
			System.out.println();
		}
	}
}
