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

// like COBPR but without separation between matrix factorization and word embedding

import java.util.HashMap;
import java.util.Map;

import com.google.common.cache.LoadingCache;

import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DataDAO;

//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;

import librec.data.DenseMatrix;
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
public class CABPR extends ContextRecommender {

	private DenseMatrix E;
	private Map<Integer, Integer> trainuser = new HashMap<Integer, Integer>();
	private Map<Integer, Integer> trainitem = new HashMap<Integer, Integer>();
	private double maxE = -Double.MAX_VALUE;
	private double maxP = -Double.MAX_VALUE;
	private double minE = Double.MAX_VALUE;
	private double minP = Double.MAX_VALUE;
	
	public CABPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();
		E = new DenseMatrix(numUsers, numItemContexts);
		E.init();
		
		userCache = trainMatrix.rowCache(cacheSpec);
		
		for (int f = 0; f < numUsers; f++) {
			String userstring = rateDao.getUserId(f);
			if (userstring.startsWith("row-")) trainuser.put(f, f);
		}
		
		for (int f = 0; f < numItems; f++) {
			if (trainMatrix.getRows(f).size()==0) continue;
			trainitem.put(f, f);
		}
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {
				if (s % (numUsers*10) == 0) System.out.println("iter\t" + iter + "\tinner\t" + s);
				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

				while (true) {
					do {
						u = Randoms.uniform(numUsers);						
					} while (!trainuser.containsKey(u));
					
					SparseVector pu = userCache.get(u);

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					do {
						j = Randoms.uniform(numItems);
					} while (pu.contains(j) || !trainitem.containsKey(j));

					break;
				}

				// update parameters
				double xui = predictE(u, i);
				double xuj = predictE(u, j);
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
				}
				
				while (true) {
					u = Randoms.uniform(numUsers);			
					SparseVector pu = userCache.get(u);

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					do {
						j = Randoms.uniform(numItems);
					} while (pu.contains(j));

					break;
				}

				// update parameters
				xui = predictP(u, i);
				xuj = predictP(u, j);
				xuij = xui - xuj;

				vals = -Math.log(g(xuij));
				loss += vals;

				cmg = g(-xuij);
				
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);
					
					P.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					Q.add(i, f, lRate * (cmg * puf - regI * qif));
					Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));
					
					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}	
			}
			
			if (isConverged(iter))
				break;
		}
		
		
		for (int u = 0; u < numUsers; u++) {
			System.out.println("computing min-max\t" + u + " of " + numUsers);
			String userstring = rateDao.getUserId(u);
			if (userstring.startsWith("row-")) {
				for (int j = 0; j < numItems; j++) {
					double scoreE = predictE(u, j);
					double scoreP = predictP(u, j);
					maxE = Math.max(maxE, scoreE);
					maxP = Math.max(maxP, scoreP);
					minE = Math.min(minE, scoreE);
					minP = Math.min(minP, scoreP);
				}
			}
		}
		
		
	}
	
	protected double predict(int u, int j) {
		double scoreP = (DenseMatrix.rowMult(P, u, Q, j)-minP)/(maxP-minP);
		double scoreE = (DenseMatrix.rowMult(E, u, itemContexts, j)-minE)/(maxE-minE);
		double val = scoreP + scoreE;
		return val;	
	}
	
	protected double predictP(int u, int j) {
		return DenseMatrix.rowMult(P, u, Q, j);			
	}
	
	protected double predictE(int u, int j) {
		if (j < itemContexts.numRows()) {
			return DenseMatrix.rowMult(E, u, itemContexts, j);				
		} else {
			return 0.0;
		}
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regC, numIters });
	}
}
