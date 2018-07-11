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

// Using normalized Image Vector

import java.util.HashMap;
import java.util.Map;

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
public class INNBPRE extends ContextRecommender {
	private Map<Integer, Map<Integer, Double>> weights = new HashMap<Integer, Map<Integer, Double>>();
	private String mappingfile;
	private Map<String, Integer> englishitemstring = new HashMap<String, Integer>();
	private Map<String, Integer> foreignitemstring = new HashMap<String, Integer>();
	private ArrayList<Integer> wusers = new ArrayList<Integer>();
	private Map<String, Integer> englishitem = new HashMap<String, Integer>();
	private Map<String, Integer> foreignitem = new HashMap<String, Integer>();
	private DenseMatrix IE;
	private DenseVector IB;
	private DenseMatrix imageContexts;
	private int numImages, numImageContexts;
	private Map<Integer, Map<Integer, Integer>> testimagemapping = new HashMap<Integer, Map<Integer, Integer>>();
	private DataDAO imageDao;
	public INNBPRE(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
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
			englishitem.put(userstring, f);
		}
		for (int f = 0; f < numItems; f++) {
			String itemstring = rateDao.getItemId(f);
			if (itemstring.startsWith("row-")) {
				englishitemstring.put(itemstring, f);
			} else {
				if (trainMatrix.getRows(f).size() > 0) {
					foreignitem.put(itemstring, f);
				}
				foreignitemstring.put(itemstring, f);
			}
		}
		mappingfile = cf.getString("mappingfile");
		contextDao = null;
		itemContexts = null;
		Runtime.getRuntime().gc();
	}

	@Override
	protected void buildModel() throws Exception {
		imageDao = new DataDAO(cf.getPath("imagefile"));
		imageContexts = imageDao.readDataDense();
		numImages = imageDao.numItems();
		numImageContexts = imageContexts.numColumns();
		Logs.debug("Read image contexts with number of items " + numImages + " and number of context columns " + numImageContexts);
		weights.clear();
		
		try {
			BufferedReader bfr = new BufferedReader(new FileReader(mappingfile));		
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.split("\t");
				String word = temp[0].trim();
				String imageword = temp[1].trim();
				if (englishitem.get(word)!=null) {
					int user = englishitem.get(word);
					Map<Integer, Double> map = weights.get(user);
					if (map == null) map = new HashMap<Integer, Double>();
					int imageitem = imageDao.getItemId(imageword);
					map.put(imageitem, (double) 1);
					weights.put(user, map);	
				} else {
					if (foreignitemstring.get(word)!=null) {
						int item = foreignitemstring.get(word);
						Map<Integer, Integer> map = testimagemapping.get(item);
						if (map == null) map = new HashMap<Integer, Integer>();
						int imageitem = imageDao.getItemId(imageword);
						map.put(imageitem, imageitem);
						testimagemapping.put(item, map);	
					}
				}
			} 
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		int numberrows = weights.size();
		for (Map.Entry<Integer, Map<Integer, Double>> entry : weights.entrySet()) {
			wusers.add(entry.getKey());
		}
		
		IE = new DenseMatrix(numberrows, numImageContexts);
		IE.init();
		IB = new DenseVector(numImageContexts);
		IB.init();
		
		lRate = 0.01;
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			for (int s = 0, smax = numberrows * 300; s < smax; s++) {
				if (s % (numberrows*10) == 0) System.out.println("iter\t" + iter + "\tinner\t" + s);
				
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(numberrows);
					int user = wusers.get(u);
					if (weights.get(user)==null) continue;
					ArrayList<Integer> pu = new ArrayList<Integer>();
					for (Map.Entry<Integer, Double> e : weights.get(user).entrySet()) {
						pu.add(e.getKey());
					}
					i = pu.get(Randoms.uniform(pu.size()));
					do {
						j = Randoms.uniform(numImages);
					}  while (pu.contains(j));

					break;
				}

				// update parameters
				double xui = predictI(u, i);
				double xuj = predictI(u, j);
				double xuij = xui - xuj;
				
				double vals = -Math.log(g(xuij));
				if (Double.isInfinite(vals)) {
					System.out.println("INFINITE!\t" + xui + "\t" + xuj + "\t" + xuij + "\t" + g(xuij) + "\t" + vals);
					continue;
				}
				loss += vals;

				double cmg = g(-xuij);

				for (int f = 0; f < numImageContexts; f++) {
					double euf = IE.get(u, f);
					double cif = imageContexts.get(i, f);
					double cjf = imageContexts.get(j, f);
					
					IE.add(u, f, lRate * (cmg * (cif - cjf) - regC * euf));

					loss += regC * euf * euf;
					
					double bf = IB.get(f);
					IB.add(f, lRate * (cmg * (cif - cjf) - regC * bf));
					
					loss += regC * bf * bf;
				}
			}
			if (isConverged(iter))
				break;
		}
		
		
	}

	protected double predictI(int u, int j) {
		return DenseMatrix.rowMult(IE, u, imageContexts, j) + IB.inner(imageContexts.row(j));
	}
	
	protected double predict(int u, int j) {
		double imageval = 0;
		if (wusers.contains(u) && testimagemapping.get(j)!=null) {
			int uid = wusers.indexOf(u);
			Map<Integer, Integer> imageitems = testimagemapping.get(j);
			for (Map.Entry<Integer, Integer> e2 : imageitems.entrySet()) {
				int imageitem = e2.getKey();
				double val = DenseMatrix.rowMult(IE, uid, imageContexts, imageitem) + IB.inner(imageContexts.row(imageitem));
				imageval = imageval + val;
			}
			imageval = imageval / (double) imageitems.size();
		}
		return imageval;
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regC, numIters });
	}
}
