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
import java.util.Collections;
import java.util.Comparator;

// Vulic's style NNBPR

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DataDAO;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.intf.IterativeRecommender;

/**
 * Context-based Bayesian Personalized Ranking (CBPR)
 * 
 * @author guoguibing
 * 
 */
public class contextOnlyBPRImages extends IterativeRecommender {
	private Map<Integer, Map<Integer, Double>> weights = new HashMap<Integer, Map<Integer, Double>>();
	private Map<String, Integer> eitemstring = new HashMap<String, Integer>();
	private Map<String, Integer> fitemstring = new HashMap<String, Integer>();
	private ArrayList<Integer> cusers = new ArrayList<Integer>();
	private Map<String, Integer> eitem = new HashMap<String, Integer>(), fitem = new HashMap<String, Integer>();
	private DenseMatrix E;
	private DenseVector B;
	private Map<Integer, Map<Integer, Double>> Escores = new HashMap<Integer, Map<Integer, Double>>();
	private DenseMatrix itemContexts;
	private int numContexts, numItemContexts;
	private DataDAO contextDao;
	
	public contextOnlyBPRImages(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
		super.initModel();
		userCache = trainMatrix.columnCache(cacheSpec);
		for (int f = 0; f < numUsers; f++) {
			String userstring = rateDao.getUserId(f);
			eitem.put(userstring, f);
		}
		for (int f = 0; f < numItems; f++) {
			String itemString = rateDao.getItemId(f);
			if (itemString.startsWith("column-")) {
				fitemstring.put(itemString, f);
			}
		}
	}

	@Override
	protected void buildModel() throws Exception {
		Map<Integer, Integer> testset = new HashMap<Integer, Integer>();
		for (int u = 0, um = testMatrix.numRows(); u < um; u++) {
			List<Integer> testItems = testMatrix.getColumns(u);
			if (testItems == null || testItems.size() == 0) continue;
			for (int i : testItems) {
				testset.put(i, i);
			}
		}
		
		String contextPath = cf.getPath("dataset.context");
		Logs.debug("Context dataset: {}", Strings.last(contextPath, 38));
		contextDao = new DataDAO(contextPath);
		itemContexts = contextDao.readDataDense();
		numContexts = contextDao.numItems();
		numItemContexts = itemContexts.numColumns();
		Logs.debug("Read item contexts with number of items " + numContexts + " and number of context columns " + numItemContexts);
		
		weights.clear();
		for (int f = 0; f < numItemContexts; f++) {
			DenseVector v = itemContexts.column(f);
			double mean = v.mean();
			v = v.minus(mean);
			double sd = 0.0;
			for (int g = 0; g < numContexts; g++) {
				sd = sd + (v.get(g)*v.get(g));
			}
			sd = sd / (double) numContexts;
			sd = Math.sqrt(sd);
			v = v.scale((double) 1 / sd);
			for (int g = 0; g < numContexts; g++) {
				itemContexts.set(g, f, v.get(g));
			}
		}
		for (int f = 0; f < numContexts; f++) {
			String itemstring = contextDao.getItemId(f);
			if (itemstring.startsWith("row-") && eitem.get(itemstring)!=null) {
				eitemstring.put(itemstring, f);
			}
			if (itemstring.startsWith("column-") && fitemstring.get(itemstring)!=null) {
				fitem.put(itemstring, f);
			}
			DenseVector v = itemContexts.row(f);
			double sd = 0.0;
			for (int g = 0; g < numItemContexts; g++) {
				sd = sd + (v.get(g)*v.get(g));
			}
			sd = Math.sqrt(sd);
			v = v.scale((double) 1 / sd);
			itemContexts.setRow(f, v);
		}

		for (int u = 0; u < numUsers; u++) {
			String userstring = rateDao.getUserId(u);
			if (eitemstring.get(userstring)!=null) {
				int uu = eitemstring.get(userstring);
				Map<Integer, Double> map = weights.get(u);
				if (map == null) map = new HashMap<Integer, Double>();
				map.put(uu, (double) 1.0);
				weights.put(u, map);
			}
		}
		int numberrows = weights.size();
		for (Map.Entry<Integer, Map<Integer, Double>> entry : weights.entrySet()) {
			cusers.add(entry.getKey());
		}
		
		E = new DenseMatrix(numberrows, numItemContexts);
		E.init();
		B = new DenseVector(numItemContexts);
		B.init();
		
		lRate = 0.01;
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (int s = 0, smax = numberrows * 300; s < smax; s++) {
				if (s % (numberrows*10) == 0) System.out.println("iter\t" + iter + "\tinner\t" + s);
				
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(numberrows);
					int user = cusers.get(u);
					if (weights.get(user)==null) continue;
					ArrayList<Integer> pu = new ArrayList<Integer>();
					for (Map.Entry<Integer, Double> e : weights.get(user).entrySet()) {
						pu.add(e.getKey());
					}
					i = pu.get(Randoms.uniform(pu.size()));
					do {
						j = Randoms.uniform(numContexts);
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
					
					E.add(u, f, lRate * (cmg * (cif - cjf) - regU * euf));

					loss += regU * euf * euf;
					
					double bf = B.get(f);
					B.add(f, lRate * (cmg * (cif - cjf) - regU * bf));
					
					loss += regU * bf * bf;
				}
				
				
			}
			
			if (isConverged(iter))
				break;
		}
		
		System.out.println("Compute maxE and minE");
		for (int u = 0; u < numUsers; u++) {
			if (u % 1000 == 0) {
				double percent = (double) u * (double) 100/(double) numUsers;
				System.out.println("Compute maxE and minE " + percent + "%");
			}
			int us = cusers.indexOf(u);
			if (us < 0) continue;
			for (Map.Entry<Integer, Integer> fi : testset.entrySet()) {
				int j = fi.getValue();
				String itemString = rateDao.getItemId(j);
				if (fitem.get(itemString)==null) continue;
				int jj = fitem.get(itemString);
				double val = DenseMatrix.rowMult(E, us, itemContexts, jj)  + B.inner(itemContexts.row(jj));
				Map<Integer, Double> currentscores = Escores.get(j);
				if (currentscores == null) currentscores = new HashMap<Integer, Double>();
				if (currentscores.size() < 100) {
					currentscores.put(u, val);
				} else {
					Vector<Integer> sorted = sortMapDouble(currentscores);
					int lastu = sorted.get(sorted.size()-1);
					double lastscore = currentscores.get(lastu);
					if (lastscore < val) {
						currentscores.remove(lastu);
						currentscores.put(u, val);
					} else if (lastscore == val) {
						currentscores.put(u, val);
					}
				}
				Escores.put(j, currentscores);
			}
		}
		
		for (Map.Entry<Integer, Map<Integer, Double>> e : Escores.entrySet()) {
			int j = e.getKey();
			String item = rateDao.getItemId(j);
			Map<Integer, Double> val = e.getValue();
			for (Map.Entry<Integer, Double> e1 : val.entrySet()) {
				int u = e1.getKey();
				String user = rateDao.getUserId(u);
				System.out.println("ESCORES\t" + item + "\t" + user + "\t" + e1.getValue());
			}
		}
		
		
	}
	protected double predictT(int u, int j) {
		return DenseMatrix.rowMult(E, u, itemContexts, j) + B.inner(itemContexts.row(j));
	}
	
	
	protected double predict(int u, int j) {
		double embeddingval = 0;
		if (Escores.get(j) != null) {
			if (Escores.get(j).get(u)!=null) {
				embeddingval = Escores.get(j).get(u);
			}
		}
		
		return embeddingval;
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regU, numIters });
	}
	
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static Vector<Integer> sortMapDouble(Map<Integer, Double> map) {
		Map<Integer, Integer> sortedResults = sortByValueDouble(map);
		List sortedQueries = new LinkedList(sortedResults.entrySet());
		Vector<Integer> sortedYears = new Vector<Integer>();
		
		for (Iterator itt = sortedQueries.iterator(); itt.hasNext();) {
			Map.Entry entry = (Map.Entry) itt.next();
			sortedYears.add(0, (Integer) entry.getKey());
		}
		return sortedYears;
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	private static Map sortByValueDouble(Map<Integer, Double> map) {
		List list = new LinkedList(map.entrySet());
		Collections.sort(list, new Comparator() {
			public int compare(Object o1, Object o2) {
				return ((Comparable) ((Map.Entry) (o1)).getValue()).compareTo(((Map.Entry) (o2)).getValue());
			}
		});
		Map result = new LinkedHashMap();
		for (Iterator it = list.iterator(); it.hasNext();) {
		     Map.Entry entry = (Map.Entry)it.next();
		     result.put(entry.getKey(), entry.getValue());
		     }
		return result;
	}
}
