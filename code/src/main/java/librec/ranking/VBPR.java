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

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

// Vulic's style NNBPR

import java.util.HashMap;
import java.util.Map;

import happy.coding.io.Strings;
import happy.coding.math.Randoms;

//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;

import librec.data.DenseMatrix;
import librec.data.DenseVector;
//import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
//import librec.data.VectorEntry;
import librec.intf.ContextRecommender;

/**
 * Context-based Bayesian Personalized Ranking (CBPR)
 * 
 * @author guoguibing
 * 
 */
public class VBPR extends ContextRecommender {
	private Map<Integer, Map<Integer, Double>> weights = new HashMap<Integer, Map<Integer, Double>>();
	private DenseMatrix M;
	private Map<Integer, Integer> foreignitem = new HashMap<Integer, Integer>();
	private Map<String, Integer> englishitemstring = new HashMap<String, Integer>();
	private DenseMatrix E;
	private DenseVector B;
	private Map<Integer, Integer> trainitem = new HashMap<Integer, Integer>();
	
	private int numItemsUsers = 0;
	
	public VBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();
		M = new DenseMatrix(numItemContexts, numItemContexts);
		M.init();
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
		numItemsUsers = foreignitem.size();
	}

	@Override
	protected void buildModel() throws Exception {

		double last_loss = 0;
		for (int iter = 1; iter <= numIters*100; iter++) {
			loss = 0;
			
			for (Map.Entry<Integer, Integer> e : foreignitem.entrySet()) {
				int user = e.getKey();
				DenseVector vu = itemContexts.row(user);
				SparseVector pu = userCache.get(user);
				int[] is = pu.getIndex();
				for (int i = 0; i < is.length; i++) {
					int item = is[i];
					DenseVector vi = itemContexts.row(englishitemstring.get(rateDao.getUserId(item)));
					for (int f = 0; f < numItemContexts; f++) {
						DenseVector v = M.row(f);
						double htheta = v.inner(vi);
						double yf = vu.get(f);
						double diff = yf - htheta;
						double pow = Math.pow(diff, 2);
						if (Double.isInfinite(pow) || Double.isNaN(pow)) {
							
						} else {
							loss += Math.pow(diff, (double) 2);
							for (int j = 0; j < numItemContexts; j++) {
								M.add(f, j, lRate * diff * vi.get(j));
							}							
						}
					}
				}
			}
			
			double diffloss = loss - last_loss;
			System.out.println(iter + "\t" + diffloss);
			if (Math.abs(diffloss) < 1e-5) break;
			else {last_loss = loss;}
		}
		
		for (Map.Entry<String, Integer> e : englishitemstring.entrySet()) {
			int j = e.getValue();
			DenseVector proj = M.mult(itemContexts.row(j));
			itemContexts.setRow(j, proj);
		}
		
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
		/*System.out.println("add ppdb");
		try {
			BufferedReader bfr = new BufferedReader(new FileReader("/nlp/users/derry/bli/ppdb.txt"));		
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.split(" \\|\\|\\| ");
				String w1 = "row-" + temp[0].trim();
				String w2 = "row-" + temp[1].trim();
				if (englishitemstring.get(w1)!=null && englishitemstring.get(w2)!=null) {
					int english1 = englishitemstring.get(w1);
					int english2 = englishitemstring.get(w2);
					
					Map<Integer, Double> map = weights.get(english1);
					if (map == null) map = new HashMap<Integer, Double>();
					map.put(english2, (double) 1.0);
					weights.put(english1, map);
					
					map = weights.get(english2);
					if (map == null) map = new HashMap<Integer, Double>();
					map.put(english1, (double) 1.0);
					weights.put(english2, map);
				}
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}*/
		
		/*userCache = trainMatrix.rowCache(cacheSpec);
		for (int u = 0; u < numUsers; u++) {
			SparseVector pu = userCache.get(u);
			if (pu.getCount() == 0)
				continue;
			int[] is = pu.getIndex();
			Map<Integer, Double> map = weights.get(u);
			if (map == null) map = new HashMap<Integer, Double>();
			for (int i : is) {
				map.put(i, (double) 1.0);
			}
			weights.put(u, map);
		}*/
		
		for (int f = 0; f < numItems; f++) {
			if (trainMatrix.getRows(f).size()==0) continue;
			trainitem.put(f, f);
		}
		//userCache = trainMatrix.rowCache(cacheSpec);
		
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {
				if (s % (numUsers*10) == 0) System.out.println("iter\t" + iter + "\tinner\t" + s);
				
				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

				/*while (true) {
					u = Randoms.uniform(numUsers);	
					
					SparseVector pu = userCache.get(u);

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					do {
						j = Randoms.uniform(numItems);
					} while (pu.contains(j) || !trainitem.containsKey(j));

					break;
				}*/
				
				while (true) {
					u = Randoms.uniform(numUsers);
					if (weights.get(u)==null) continue;
					ArrayList<Integer> pu = new ArrayList<Integer>();
					for (Map.Entry<Integer, Double> e : weights.get(u).entrySet()) {
						pu.add(e.getKey());
					}
					i = pu.get(Randoms.uniform(pu.size()));
					//SparseVector puu = userCache.get(u);
					do {
						j = Randoms.uniform(numItems);
					}  while (pu.contains(j));// || puu.contains(j));

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
	
	protected double predictM(int english, int foreign) {
		String englishstring = rateDao.getUserId(english);
		DenseVector vu = itemContexts.row(englishitemstring.get(englishstring));
		DenseVector vj = itemContexts.row(foreign);
		DenseVector proj = M.mult(vu);
		double suma = 0.0;
		double sumb = 0.0;
		for (int g = 0; g < numItemContexts; g++) {
			suma = suma + Math.pow(proj.get(g), (double) 2);
			sumb = sumb + Math.pow(vj.get(g), (double) 2);
		}
		double denom = Math.sqrt(suma) * Math.sqrt(sumb);
		double val = vj.inner(proj);
		if (denom > 0) {
			val = val / denom;
		}
		return val;
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regC, numIters });
	}
}
