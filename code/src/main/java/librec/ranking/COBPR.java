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

import java.util.HashMap;
import java.util.Map;

import happy.coding.io.Strings;
import happy.coding.math.Randoms;

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

// BPR with matrix factorization and additional learned matrix (E) for signal from word embedding
/**
 * Context-based Bayesian Personalized Ranking (CBPR)
 * 
 * @author guoguibing
 * 
 */
public class COBPR extends ContextRecommender {

	private DenseMatrix E;//, EE, PP, QQ;
	private Map<Integer, Integer> trainuser = new HashMap<Integer, Integer>();
	
	public COBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();
		
		E = new DenseMatrix(numUsers, numItemContexts);
		//EE = new DenseMatrix(numUsers, 2);
		E.init();
		//EE.init();
		
		/*PP = new DenseMatrix(numUsers, numFactors);
		QQ = new DenseMatrix(numItems, numFactors);

		// initialize model
		if (initByNorm) {
			PP.init(initMean, initStd);
			QQ.init(initMean, initStd);
		} else {
			PP.init(); // P.init(smallValue);
			QQ.init(); // Q.init(smallValue);
		}*/

		userCache = trainMatrix.rowCache(cacheSpec);
		
		for (int f = 0; f < numUsers; f++) {
			String userstring = rateDao.getUserId(f);
			if (userstring.startsWith("row-")) trainuser.put(f, f);
		}
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

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
				double xui = predict(u, i);
				double xuj = predict(u, j);
				double xuij = xui - xuj;

				double vals = -Math.log(g(xuij));
				loss += vals;

				double cmg = g(-xuij);
				
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);
					
					P.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					Q.add(i, f, lRate * (cmg * puf - regI * qif));
					Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));
					
					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}
				
				if (!trainuser.containsKey(u)) continue;
				for (int f = 0; f < numItemContexts; f++) {
					double euf = E.get(u, f);
					double cif = itemContexts.get(i, f);
					double cjf = itemContexts.get(j, f);
					
					E.add(u, f, lRate * (cmg * (cif - cjf) - regC * euf));

					loss += regC * euf * euf;
				}
				
			}

			if (isConverged(iter))
				break;

		}
	}
	
	/*@Override
	protected void buildModel() throws Exception {
		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

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
				double xui = predictT(u, i);
				double xuj = predictT(u, j);
				double xuij = xui - xuj;

				double vals = -Math.log(g(xuij));
				loss += vals;

				double cmg = g(-xuij);
				
				for (int f = 0; f < numFactors; f++) {
					double puf = PP.get(u, f);
					double qif = QQ.get(i, f);
					double qjf = QQ.get(j, f);
					
					PP.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					QQ.add(i, f, lRate * (cmg * puf - regI * qif));
					QQ.add(j, f, lRate * (cmg * (-puf) - regI * qjf));
					
					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}
				
				if (!trainuser.containsKey(u)) continue;
				for (int f = 0; f < numItemContexts; f++) {
					double euf = E.get(u, f);
					double cif = itemContexts.get(i, f);
					double cjf = itemContexts.get(j, f);
					
					E.add(u, f, lRate * (cmg * (cif - cjf) - regC * euf));

					loss += regC * euf * euf;
				}
				
			}

			if (isConverged(iter))
				break;

		}
		
		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

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
				double xui = predictTT(u, i);
				double xuj = predictTT(u, j);
				double xuij = xui - xuj;

				double vals = -Math.log(g(xuij));
				loss += vals;

				double cmg = g(-xuij);
				
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
		
		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			for (int s = 0, smax = numUsers * 300; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

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
				double xui = predict(u, i);
				double xuj = predict(u, j);
				double xuij = xui - xuj;

				double vals = -Math.log(g(xuij));
				loss += vals;

				double cmg = g(-xuij);
				
				for (int f = 0; f < 2; f++) {
					double euf = EE.get(u, f);
					double cif = 0, cjf = 0;
					if (f == 0) {
						cif = predictT(u, i);
						cjf = predictT(u, j);
					} else {
						cif = predictTT(u, i);
						cjf = predictTT(u, j);
					}
					
					EE.add(u, f, lRate * (cmg * (cif - cjf) - regC * euf));

					loss += regC * euf * euf;
				}
				
			}

			if (isConverged(iter))
				break;

		}
		
		

	}
	
	protected double predict(int u, int j) {
		double coeff1 = EE.get(u, 0);
		double coeff2 = EE.get(u, 1);
		double val = coeff1 * predictT(u, j) + coeff2 * predictTT(u, j);
		return val;
	}
	
	protected double predictT(int u, int j) {
		//return DenseMatrix.rowMult(E, u, itemContexts, j);
		if (!trainuser.containsKey(u)) {
			return DenseMatrix.rowMult(PP, u, QQ, j);
		} else {
			return DenseMatrix.rowMult(PP, u, QQ, j) + DenseMatrix.rowMult(E, u, itemContexts, j);			
		}
	}
	
	protected double predictTT(int u, int j) {
		return DenseMatrix.rowMult(P, u, Q, j);
		if (!trainuser.containsKey(u)) {
			return DenseMatrix.rowMult(P, u, Q, j);
		} else {
			return DenseMatrix.rowMult(P, u, Q, j) + DenseMatrix.rowMult(E, u, itemContexts, j);			
		}
	}*/

	protected double predict(int u, int j) {
		return DenseMatrix.rowMult(P, u, Q, j) + DenseMatrix.rowMult(E, u, itemContexts, j);			
		/*if (!trainuser.containsKey(u)) {
			return DenseMatrix.rowMult(P, u, Q, j);
		} else {
			return DenseMatrix.rowMult(P, u, Q, j) + DenseMatrix.rowMult(E, u, itemContexts, j);			
		}*/
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regC, numIters });
	}
	/*
	private double getMean(ArrayList<Double> data)
    {
        double sum = 0.0;
        for(double a : data)
            sum += a;
        return sum/data.size();
    }

    private double getVariance(ArrayList<Double> data, double mean)
    {
        double temp = 0;
        for(double a :data)
            temp += (a-mean)*(a-mean);
        return temp/data.size();
    }

    private double getStdDev(ArrayList<Double> data, double mean)
    {
        return Math.sqrt(getVariance(data, mean));
    }
*/
}
