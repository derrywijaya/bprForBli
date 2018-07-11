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
public class VNBPR extends ContextRecommender {

	private DenseMatrix E;
	private DenseVector B;
	private Map<Integer, Integer> foreignitem = new HashMap<Integer, Integer>();
	private Map<String, Integer> englishitemstring = new HashMap<String, Integer>();
	
	private int numItemsUsers = 0;
	
	public VNBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();
		E = new DenseMatrix(numItemContexts, numItemContexts);
		E.init();
		B = new DenseVector(numItemContexts);
		B.init();
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
		for (int iter = 1; iter <= numIters; iter++) {
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
						DenseVector v = E.row(f);
						double htheta = v.inner(vi);
						double intercept = B.get(f);
						htheta = htheta + intercept;
						double yf = vu.get(f);
						double diff = yf - htheta;
						double pow = Math.pow(diff, 2);
						if (Double.isInfinite(pow) || Double.isNaN(pow)) {
							
						} else {
							loss += Math.pow(diff, (double) 2);
							for (int j = 0; j < numItemContexts; j++) {
								E.add(f, j, lRate * diff * vi.get(j));
							}
							double newintercept = B.get(f) + (lRate * diff);
							B.set(f, newintercept);
						}
					}
				}
			}
			
			double diffloss = loss - last_loss;
			System.out.println(iter + "\t" + diffloss);
			if (Math.abs(diffloss) < 1e-5) break;
			else {last_loss = loss;}
		}
	}

	
	protected double predict(int english, int foreign) {
		String englishstring = rateDao.getUserId(english);
		DenseVector vu = itemContexts.row(englishitemstring.get(englishstring));
		DenseVector vj = itemContexts.row(foreign);
		DenseVector proj = E.mult(vu);
		proj = proj.add(B);
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
