package afterphd;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.core.Utils;
import afc.basic.statistics.ConfidenceInterval;
/*
 * The purpose of this class is to check again if the following is true.
 * 1) The chance of getting more good samples than original sample are higher when good samples are more than 50%
 */
public class TestSamplingWithReplacement {
	public static void main(String[] args) {
		final int sampleSize = 100;
		final int bootstrapReplicates = 1000;
		final int repeats = 1000;
		Random rand2 = new Random(922);
		// *** Run from 0 to 1
		for (int goodSampleRatio = 0; goodSampleRatio <= 10; goodSampleRatio += 1) {
			final int goodSampleSize = goodSampleRatio * sampleSize / 10;
			System.out.println(String.format("Good Sample Size: %s", goodSampleSize));
			int totalGoodReplicates = 0;
			List<Double> dList = new ArrayList<Double>();
			for (int y = 0; y < repeats; y++) {
				Random rand = new Random(rand2.nextLong());
				// *** Number of replicates
				int betterSample = 0;
				int worseSample = 0;
				for (int r = 0; r < bootstrapReplicates; r++) {
					// *** Sampling with replacement
					int goodCount = 0;
					for (int i = 0; i < sampleSize; i++) {
						int randInt = rand.nextInt(sampleSize);
						if (randInt < goodSampleSize) goodCount++;
					}
					if (goodCount > goodSampleSize) betterSample++;
					if (goodCount < goodSampleSize) worseSample++;
				}
				if (betterSample > worseSample) totalGoodReplicates++;
				dList.add(betterSample * 1.0 / bootstrapReplicates);
			}
			double[] ci = ConfidenceInterval.compute(dList, 0.99);
			double upper = Utils.roundDouble(ci[0] + ci[1], 2);
			double lower = Utils.roundDouble(ci[0] - ci[1], 2);
			System.out.println("Range: " + lower + ", " + upper);
			System.out.println("TotalGoodReplicates: " + totalGoodReplicates);
			System.out.println();
		}
	}
}
