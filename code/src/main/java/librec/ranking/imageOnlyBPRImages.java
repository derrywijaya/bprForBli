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
public class imageOnlyBPRImages extends IterativeRecommender {
	private Map<Integer, Map<Integer, Double>> weights = new HashMap<Integer, Map<Integer, Double>>();
	private String lang;
	private String mappingfile;
	private Map<String, Integer> fitemstring = new HashMap<String, Integer>();
	private ArrayList<Integer> wusers = new ArrayList<Integer>();
	private Map<String, Integer> eitem = new HashMap<String, Integer>();
	private DenseMatrix IE;
	private DenseVector IB;
	private Map<Integer, Map<Integer, Double>> Iscores = new HashMap<Integer, Map<Integer, Double>>();
	private DenseMatrix imageContexts;
	private int numImages, numImageContexts;
	private Map<Integer, Map<Integer, Integer>> testimagemapping = new HashMap<Integer, Map<Integer, Integer>>();
	private DataDAO imageDao;
	
	public imageOnlyBPRImages(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
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
		lang = cf.getString("mainlanguage");
		if (lang.equalsIgnoreCase("eb")) lang = "ceb";
		mappingfile = cf.getString("mappingfile");
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
				if (eitem.get(word)!=null) {
					int user = eitem.get(word);
					Map<Integer, Double> map = weights.get(user);
					if (map == null) map = new HashMap<Integer, Double>();
					int imageitem = imageDao.getItemId(imageword);
					map.put(imageitem, (double) 1);
					weights.put(user, map);	
				} else {
					if (fitemstring.get(word)!=null) {
						int item = fitemstring.get(word);
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
					
					IE.add(u, f, lRate * (cmg * (cif - cjf) - regU * euf));

					loss += regU * euf * euf;
					
					double bf = IB.get(f);
					IB.add(f, lRate * (cmg * (cif - cjf) - regU * bf));
					
					loss += regU * bf * bf;
				}
			}
			if (isConverged(iter))
				break;
		}
		
		System.out.println("Compute maxI and minI");
		for (int u = 0; u < numUsers; u++) {
			if (u % 1000 == 0) {
				double percent = (double) u * (double) 100/(double) numUsers;
				System.out.println("Compute maxI and minI " + percent + "%");
			}
			if (!wusers.contains(u)) continue;
			int uid = wusers.indexOf(u);
			for (Map.Entry<Integer, Integer> fi : testset.entrySet()) {
				int j = fi.getValue();
				if (testimagemapping.get(j)!=null) {
					double imageval = 0;
					Map<Integer, Integer> imageitems = testimagemapping.get(j);
					for (Map.Entry<Integer, Integer> e2 : imageitems.entrySet()) {
						int imageitem = e2.getKey();
						double val = DenseMatrix.rowMult(IE, uid, imageContexts, imageitem) + IB.inner(imageContexts.row(imageitem));
						imageval = imageval + val;
					}
					imageval = imageval / (double) imageitems.size();
					Map<Integer, Double> currentscores = Iscores.get(j);
					if (currentscores == null) currentscores = new HashMap<Integer, Double>();
					if (currentscores.size() < 100) {
						currentscores.put(u, imageval);
					} else {
						Vector<Integer> sorted = sortMapDouble(currentscores);
						int lastu = sorted.get(sorted.size()-1);
						double lastscore = currentscores.get(lastu);
						if (lastscore < imageval) {
							currentscores.remove(lastu);
							currentscores.put(u, imageval);
						} else if (lastscore == imageval) {
							currentscores.put(u, imageval);
						}
					}
					//currentscores.put(u, imageval);
					Iscores.put(j, currentscores);
				}
			}
		}
		
		for (Map.Entry<Integer, Map<Integer, Double>> e : Iscores.entrySet()) {
			int j = e.getKey();
			String item = rateDao.getItemId(j);
			Map<Integer, Double> val = e.getValue();
			for (Map.Entry<Integer, Double> e1 : val.entrySet()) {
				int u = e1.getKey();
				String user = rateDao.getUserId(u);
				System.out.println("ISCORES\t" + item + "\t" + user + "\t" + e1.getValue());
			}
		}
	}
	protected double predictI(int u, int j) {
		return DenseMatrix.rowMult(IE, u, imageContexts, j) + IB.inner(imageContexts.row(j));
	}
	
	protected double predict(int u, int j) {
		double imageval = 0;
		if (Iscores.get(j) != null) {
			if (Iscores.get(j).get(u)!=null) {
				imageval = Iscores.get(j).get(u);
			}
		}

		return imageval;
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
