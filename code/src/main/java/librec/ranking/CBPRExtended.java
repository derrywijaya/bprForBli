// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.ranking;

import java.util.ArrayList;

// Vulic's style NNBPR

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.AbstractMap.SimpleImmutableEntry;

import happy.coding.io.FileIO;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DataDAO;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.intf.ContextRecommender;

/**
 * Context-based Bayesian Personalized Ranking (CBPR)
 * 
 * @author guoguibing
 * 
 */
public class CBPRExtended extends ContextRecommender {
	private Map<Integer, Map<Integer, Double>> weights = new HashMap<Integer, Map<Integer, Double>>();
	private Map<Integer, Integer> foreignitem = new HashMap<Integer, Integer>();
	private Map<String, Integer> englishitemstring = new HashMap<String, Integer>();
	private DenseMatrix E;
	private DenseVector B;
	private Map<Integer, Integer> trainitem = new HashMap<Integer, Integer>();
	private DataDAO extendedDao;
	private DenseMatrix extendedItemContexts;
	
	public CBPRExtended(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();
		userCache = trainMatrix.columnCache(cacheSpec);
		for (int f = 0; f < numItems; f++) {
			String itemstring = rateDao.getItemId(f);
			if (itemstring.startsWith("row-")) {
				englishitemstring.put(itemstring, f);
			} else {
				if (trainMatrix.getRows(f).size() > 0) {
					foreignitem.put(f, f);
				}
			}
		}
		
		extendedDao = new DataDAO(cf.getPath("dataset.extended"));
		extendedItemContexts = extendedDao.readDataDense();
	}

	@Override
	protected void buildModel() throws Exception {
		for (int f = 0; f < numItemContexts; f++) {
			DenseVector v = itemContexts.column(f);
			double mean = v.mean();
			v = v.minus(mean);
			double sd = 0.0;
			for (int g = 0; g < numItems; g++) {
				sd = sd + (v.get(g)*v.get(g));
			}
			sd = sd / (double) numItems;
			sd = Math.sqrt(sd);
			v = v.scale((double) 1 / sd);
			for (int g = 0; g < numItems; g++) {
				itemContexts.set(g, f, v.get(g));
			}
		}
		
		for (int f = 0; f < numItems; f++) {
			DenseVector v = itemContexts.row(f);
			double sd = 0.0;
			for (int g = 0; g < numItemContexts; g++) {
				sd = sd + (v.get(g)*v.get(g));
			}
			sd = Math.sqrt(sd);
			v = v.scale((double) 1 / sd);
			itemContexts.setRow(f, v);
		}
		
		E = new DenseMatrix(numUsers, numItemContexts);
		E.init();
		B = new DenseVector(numItemContexts);
		B.init();
		for (int u = 0; u < numUsers; u++) {
			String userstring = rateDao.getUserId(u);
			int uu = englishitemstring.get(userstring);
			Map<Integer, Double> map = weights.get(u);
			if (map == null) map = new HashMap<Integer, Double>();
			map.put(uu, (double) 1.0);
			weights.put(u, map);
		}
		
		for (int f = 0; f < numItems; f++) {
			if (trainMatrix.getRows(f).size()==0) continue;
			trainitem.put(f, f);
		}
		
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {
				if (s % (numUsers*10) == 0) System.out.println("iter\t" + iter + "\tinner\t" + s);
				
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(numUsers);
					if (weights.get(u)==null) continue;
					ArrayList<Integer> pu = new ArrayList<Integer>();
					for (Map.Entry<Integer, Double> e : weights.get(u).entrySet()) {
						pu.add(e.getKey());
					}
					i = pu.get(Randoms.uniform(pu.size()));
					do {
						j = Randoms.uniform(numItems);
					}  while (pu.contains(j));

					break;
				}

				// update parameters
				double xui = predictT(u, i);
				double xuj = predictT(u, j);
				double xuij = xui - xuj;

				double vals = -Math.log(g(xuij));
				loss += vals;

				double cmg = g(-xuij);

				for (int f = 0; f < numItemContexts; f++) {
					double euf = E.get(u, f);
					double cif = itemContexts.get(i, f);
					double cjf = itemContexts.get(j, f);
					
					E.add(u, f, lRate * (cmg * (cif - cjf) - regC * euf));

					loss += regC * euf * euf;
					
					double bf = B.get(f);
					B.add(f, lRate * (cmg * (cif - cjf) - regC * bf));
					
					loss += regC * bf * bf;
				}
				
				
			}
			
			if (isConverged(iter))
				break;
		}
		
		List<String> preds = null;
		String toFile = null;
		if (isResultsOut) {
			preds = new ArrayList<String>(1500);
			preds.add("# userId: recommendations in (itemId, ranking score) pairs, where a correct recommendation is denoted by symbol *."); // optional: file header
			toFile = tempDirPath + algoName + "-top-10-items-extended" + foldInfo + ".txt"; // the output-file name
			FileIO.deleteFile(toFile); // delete possibly old files
		}
		
		for (int j = 0, jm = extendedItemContexts.numRows(); j < jm; j++) {
			List<Map.Entry<Integer, Double>> userScores = new ArrayList<>(Lists.initSize(numUsers));
			for (int u = 0; u < numUsers; u++) {
				double val = predictExtended(u, j);
				userScores.add(new SimpleImmutableEntry<Integer, Double>(u, val));
			}
			Lists.sortList(userScores, true);
			List<Map.Entry<Integer, Double>> recomd = (numRecs <= 0 || userScores.size() <= numRecs) ? userScores
					: userScores.subList(0, numRecs);
			List<Integer> rankedUsers = new ArrayList<>();
			StringBuilder sb = new StringBuilder();
			int count = 0;
			
			for (Map.Entry<Integer, Double> kv : recomd) {
				Integer user = kv.getKey();
				String userstring = rateDao.getUserId(user);
				
				rankedUsers.add(user);

				if (isResultsOut && count < 10) {
					// restore back to the original item id
					sb.append("(").append(userstring);
					sb.append(", ").append(kv.getValue().floatValue()).append(")");

					if (++count >= 10)
						break;
					if (count < 10)
						sb.append(", ");
				}
			}
			
			// output predictions
			if (isResultsOut) {
				// restore back to the original user id
				preds.add(j + ": " + extendedDao.getItemId(j) + ": " + sb.toString());
				if (preds.size() >= 1000) {
					FileIO.writeList(toFile, preds, true);
					preds.clear();
				}
			}
		}
		
		// write results out first
		if (isResultsOut && preds.size() > 0) {
			FileIO.writeList(toFile, preds, true);
			Logs.debug("{}{} has writeen item recommendations to {}", algoName, foldInfo, toFile);
		}		
				
	}

	protected double predictExtended(int u, int j) {
		return DenseMatrix.rowMult(E, u, extendedItemContexts, j) + B.inner(extendedItemContexts.row(j));
	}
	
	protected double predictT(int u, int j) {
		return DenseMatrix.rowMult(E, u, itemContexts, j) + B.inner(itemContexts.row(j));
	}
	
	protected double predict(int u, int j) {
		if (j < itemContexts.numRows()) {
			return DenseMatrix.rowMult(E, u, itemContexts, j)  + B.inner(itemContexts.row(j));
		} else {
			return 0.0;
		}
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regC, numIters });
	}
}
