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
import librec.intf.ContextRecommender;

/**
 * Context-based Bayesian Personalized Ranking (CBPR)
 * 
 * @author guoguibing
 * 
 */
public class AllNNBPRE extends ContextRecommender {
	private Map<Integer, Map<Integer, Double>> weights = new HashMap<Integer, Map<Integer, Double>>();
	private String lang;
	private String mappingfile;
	private Map<String, Integer> englishitemstring = new HashMap<String, Integer>();
	private Map<String, Integer> foreignitemstring = new HashMap<String, Integer>();
	private ArrayList<Integer> wusers = new ArrayList<Integer>();
	private ArrayList<Integer> wikiusers = new ArrayList<Integer>();
	private ArrayList<Integer> wikiitems = new ArrayList<Integer>();
	private ArrayList<Integer> thirdusers = new ArrayList<Integer>();
	private ArrayList<Integer> thirditems = new ArrayList<Integer>();
	private Map<Integer, Integer> countthirditems = new HashMap<Integer, Integer>();
	private Map<String, Integer> englishitem = new HashMap<String, Integer>();
	private Map<String, Integer> foreignitem = new HashMap<String, Integer>();
	private DenseMatrix E, TP, TQ, IE;
	private DenseVector B, IB;
	private Map<Integer, Integer> trainitem = new HashMap<Integer, Integer>();
	private double maxE = - Double.MAX_VALUE;
	private double minE = Double.MAX_VALUE;
	private double maxI = - Double.MAX_VALUE;
	private double minI = Double.MAX_VALUE;
	private Map<Integer, Map<Integer, Double>> Escores = new HashMap<Integer, Map<Integer, Double>>();
	private Map<Integer, Map<Integer, Double>> Iscores = new HashMap<Integer, Map<Integer, Double>>();
	private Map<Integer, Map<Integer, Double>> Wscores = new HashMap<Integer, Map<Integer, Double>>();
	private Map<Integer, Map<Integer, Double>> Tscores = new HashMap<Integer, Map<Integer, Double>>();
	private DenseMatrix imageContexts;
	private int numImages, numImageContexts;
	private Map<Integer, Map<Integer, Integer>> testimagemapping = new HashMap<Integer, Map<Integer, Integer>>();
	private DataDAO imageDao;
	
	public AllNNBPRE(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
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
		
		Map<String, Integer> wikiIndices = new HashMap<String, Integer>();
		int current = numUsers;
		
		System.out.println("read interlingua links" + " " + lang + " " + "/home1/d/derry/wikipedia/interlanguage.txt");
		try {
			BufferedReader bfr = new BufferedReader(new FileReader("/home1/d/derry/wikipedia/interlanguage.txt"));		
			String l;
			while ((l = bfr.readLine()) != null) {
				l = l.toLowerCase();
				String temp[] = l.split(" ::: ");
				String english = temp[0].trim();
				String foreign = null;
				for (int i = 1; i < temp.length; i++) {
					String t = temp[i].trim();
					String temps[] = t.split(" \\|\\|\\| ");
					String language = temps[0].trim();
					String word = temps[1].trim();
					if (language.equalsIgnoreCase(lang)) {
						foreign = word;
						break;
					}
				}
				if (foreign != null) {
					String w1 = "row-" + english;
					String wiki = "wiki-" + english;
					String w2 = "column-" + foreign;
					if (englishitemstring.get(w1)!=null && foreignitemstring.get(w2)!=null) {
						System.out.println(english + "\t" + foreign);
						
						int englishint = englishitemstring.get(w1);
						int foreignint = foreignitemstring.get(w2);
						int wikiint;
						if (wikiIndices.get(wiki)!=null) {
							wikiint = wikiIndices.get(wiki);
						} else {
							wikiint = current;
							wikiIndices.put(wiki, wikiint);
							current++;
						}
						
						int u = rateDao.getUserId(w1);
						Map<Integer, Double> map = weights.get(u);
						if (map == null) map = new HashMap<Integer, Double>();
						map.put(englishint, (double) 1.0);
						weights.put(u, map);
						
						map = weights.get(wikiint);
						if (map == null) map = new HashMap<Integer, Double>();
						map.put(englishint, (double) 1);
						map.put(foreignint, (double) 1);
						weights.put(wikiint, map);
						
					}
				}
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (Map.Entry<Integer, Map<Integer, Double>> e : weights.entrySet()) {
			wikiusers.add(e.getKey());
			Map<Integer,Double> map = e.getValue();
			for (Map.Entry<Integer, Double> e1 : map.entrySet()) {
				int j = e1.getKey();
				if (!wikiitems.contains(j)) wikiitems.add(j);
			}
		}
		
		P = new DenseMatrix(wikiusers.size(), numFactors);
		Q = new DenseMatrix(wikiitems.size(), numFactors);
		P.init();
		Q.init();
		
		
		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			for (int s = 0, smax = weights.size() * 300; s < smax; s++) {
				if (s % (numUsers*10) == 0) System.out.println("iter\t" + iter + "\tinner\t" + s);
				
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(weights.size());
					int wu = wikiusers.get(u);
					ArrayList<Integer> pu = new ArrayList<Integer>();
					for (Map.Entry<Integer, Double> e : weights.get(wu).entrySet()) {
						pu.add(e.getKey());
					}
					int wi = pu.get(Randoms.uniform(pu.size()));
					i = wikiitems.indexOf(wi);
					int wj;
					do {
						j = Randoms.uniform(wikiitems.size());
						wj = wikiitems.get(j);
					}  while (pu.contains(wj));

					break;
				}

				// update parameters
				double xui = predictP(u, i);
				
				double xuj = predictP(u, j);
				
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
		
		for (int u = 0; u < numUsers; u++) {
			if (u % 1000 == 0) {
				double percent = (double) u * (double) 100/(double) numUsers;
				System.out.println("Compute Wikiscores " + percent + "%");
			}
			if (!wikiusers.contains(u)) continue;
			int us = wikiusers.indexOf(u);
			for (Map.Entry<Integer, Integer> fi : testset.entrySet()) {
				int j = fi.getValue();
				if (wikiitems.contains(j)) {
					int jj = wikiitems.indexOf(j);
					double val = DenseMatrix.rowMult(P, us, Q, jj);
					Map<Integer, Double> currentscores = Wscores.get(j);
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
					Wscores.put(j, currentscores);
				}
			}
		}
		
		for (Map.Entry<Integer, Map<Integer, Double>> e : Wscores.entrySet()) {
			int j = e.getKey();
			String item = rateDao.getItemId(j);
			Map<Integer, Double> val = e.getValue();
			for (Map.Entry<Integer, Double> e1 : val.entrySet()) {
				int u = e1.getKey();
				String user = rateDao.getUserId(u);
				System.out.println("WSCORES\t" + item + "\t" + user + "\t" + e1.getValue());
			}
		}
		
		weights.clear();
		Map<String, Map<String,Map<String,String>>> thirdlanguages = new HashMap<String, Map<String,Map<String,String>>>();
		try {
			BufferedReader bfr = new BufferedReader(new FileReader("/nlp/users/derry/bilex/languagesdata.txt"));		
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.split(" \\|\\|\\| ");
				String language = temp[0].trim();
				if (language.equalsIgnoreCase(lang)) continue; // don't read your own translations
				String foreign = temp[1].trim();
				String english = temp[2].trim();
				String rf = "column-" + foreign.replace(" ", "_").replace(",", "_");
				String ce = "row-" + english.replace(" ", "_").replace(",", "_");
				if (englishitemstring.get(ce)!=null) {
					Map<String, Map<String,String>> map = thirdlanguages.get(language);
					if (map==null) map = new HashMap<String, Map<String,String>>();
					Map<String, String> map2 = map.get(rf);
					if (map2 == null) map2 = new HashMap<String,String>();
					map2.put(ce, ce);
					map.put(rf, map2);
					thirdlanguages.put(language, map);
				}
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		wikiIndices.clear();
		current = numUsers;
		try {
			BufferedReader bfr = new BufferedReader(new FileReader("/home1/d/derry/wikipedia/interlanguage.txt"));		
			String l;
			while ((l = bfr.readLine()) != null) {
				l = l.toLowerCase();
				String temp[] = l.split(" ::: ");
				String foreign = null;
				Map<String, String> foreign2s = new HashMap<String,String>();
				for (int i = 1; i < temp.length; i++) {
					String t = temp[i].trim();
					String temps[] = t.split(" \\|\\|\\| ");
					String language = temps[0].trim();
					String word = temps[1].trim();
					if (language.equalsIgnoreCase(lang)) {
						foreign = word;
					}
					if (thirdlanguages.get(language)!=null) {
						foreign2s.put(language, word);
					}
				}
				
				if (foreign !=null && foreign2s.size() > 0) {
					for (Map.Entry<String, String> entry : foreign2s.entrySet()) {
						String language = entry.getKey();
						String foreign2 = entry.getValue();
						Map<String, Map<String, String>> map1 = thirdlanguages.get(language);
						foreign2 = foreign2.replace(" ", "_").replace(",", "_");
						String rf = "column-" + foreign2;
						if (map1.get(rf)!=null) {
							Map<String,String> map2 = map1.get(rf);
							for (Map.Entry<String, String> entry2 : map2.entrySet()) {
								String w1 = entry2.getKey();
								String wiki = language + "-" + foreign2;
								String w2 = "column-" + foreign;
								if (englishitemstring.get(w1)!=null && foreignitemstring.get(w2)!=null) {
									System.out.println(w1 + "\t" + wiki + "\t" + foreign);
									
									int englishint = englishitemstring.get(w1);
									int foreignint = foreignitemstring.get(w2);
									int wikiint;
									if (wikiIndices.get(wiki)!=null) {
										wikiint = wikiIndices.get(wiki);
									} else {
										wikiint = current;
										wikiIndices.put(wiki, wikiint);
										current++;
									}
									
									int u = rateDao.getUserId(w1);
									Map<Integer, Double> map = weights.get(u);
									if (map == null) map = new HashMap<Integer, Double>();
									map.put(englishint, (double) 1.0);
									weights.put(u, map);
									
									map = weights.get(wikiint);
									if (map == null) map = new HashMap<Integer, Double>();
									map.put(englishint, (double) 1);
									map.put(foreignint, (double) 1);
									weights.put(wikiint, map);
									
								}
							}
						}
					}
				}
				
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		for (Map.Entry<Integer, Map<Integer, Double>> e : weights.entrySet()) {
			thirdusers.add(e.getKey());
			Map<Integer,Double> map = e.getValue();
			for (Map.Entry<Integer, Double> e1 : map.entrySet()) {
				int j = e1.getKey();
				if (!thirditems.contains(j)) thirditems.add(j);
				Integer currentcount = countthirditems.get(j);
				if (currentcount == null) currentcount = 1;
				else currentcount = currentcount + 1;
				countthirditems.put(j, currentcount);
			}
		}
		
		TP = new DenseMatrix(thirdusers.size(), numFactors);
		TQ = new DenseMatrix(thirditems.size(), numFactors);
		TP.init();
		TQ.init();
		
		lRate = 0.01;
		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			for (int s = 0, smax = weights.size() * 300; s < smax; s++) {
				if (s % (numUsers*10) == 0) System.out.println("iter\t" + iter + "\tinner\t" + s);
				
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(weights.size());
					int wu = thirdusers.get(u);
					ArrayList<Integer> pu = new ArrayList<Integer>();
					for (Map.Entry<Integer, Double> e : weights.get(wu).entrySet()) {
						pu.add(e.getKey());
					}
					int wi = pu.get(Randoms.uniform(pu.size()));
					i = thirditems.indexOf(wi);
					int wj;
					do {
						j = Randoms.uniform(thirditems.size());
						wj = thirditems.get(j);
					}  while (pu.contains(wj));

					break;
				}

				// update parameters
				double xui = predictTP(u, i);
				double xuj = predictTP(u, j);
				double xuij = xui - xuj;

				double vals = -Math.log(g(xuij));
				loss += vals;

				double cmg = g(-xuij);

				for (int f = 0; f < numFactors; f++) {
					double puf = TP.get(u, f);
					double qif = TQ.get(i, f);
					double qjf = TQ.get(j, f);

					TP.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					TQ.add(i, f, lRate * (cmg * puf - regI * qif));
					TQ.add(j, f, lRate * (cmg * (-puf) - regI * qjf));

					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}
			}

			if (isConverged(iter))
				break;

		}
		
		for (int u = 0; u < numUsers; u++) {
			if (u % 1000 == 0) {
				double percent = (double) u * (double) 100/(double) numUsers;
				System.out.println("Compute Thirdlangscores " + percent + "%");
			}
			if (!thirdusers.contains(u)) continue;
			int us = thirdusers.indexOf(u);
			for (Map.Entry<Integer, Integer> fi : testset.entrySet()) {
				int j = fi.getValue();
				if (thirditems.contains(j) && countthirditems.get(j) > 1) {
					int jj = thirditems.indexOf(j);
					double val = DenseMatrix.rowMult(TP, us, TQ, jj);
					Map<Integer, Double> currentscores = Tscores.get(j);
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
					Tscores.put(j, currentscores);
				}
			}
		}
		
		for (Map.Entry<Integer, Map<Integer, Double>> e : Tscores.entrySet()) {
			int j = e.getKey();
			String item = rateDao.getItemId(j);
			Map<Integer, Double> val = e.getValue();
			for (Map.Entry<Integer, Double> e1 : val.entrySet()) {
				int u = e1.getKey();
				String user = rateDao.getUserId(u);
				System.out.println("TSCORES\t" + item + "\t" + user + "\t" + e1.getValue());
			}
		}
		
		weights.clear();
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
		
		lRate = 0.01;
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
		
		int nr = numUsers;
		int nc = numItemContexts;

		for (int u = 0; u < nr; u++) {
			String userstring =rateDao.getUserId(u);
			String toprint = "";
			for (int j = 0; j < nc; j++) {
				toprint = toprint + E.get(u, j) + " ";
			}
			toprint = userstring + " " + toprint.trim();
			System.out.println("EMatrix " + toprint);
		}
		
		String b = "";
		for (int j = 0; j < nc; j++) {
			b = b + B.get(j) + " ";
		}
		System.out.println("BVector " + b.trim());
		
		nr = numItems;
		nc = numItemContexts;
		for (int u = 0; u < nr; u++) {
			String userstring =rateDao.getItemId(u);
			String toprint = "";
			for (int j = 0; j < nc; j++) {
				toprint = toprint + itemContexts.get(u, j) + " ";
			}
			toprint = userstring + " " + toprint.trim();
			System.out.println("itemContexts " + toprint);
		}
		
		nr = IE.numRows();
		nc = IE.numColumns();

		for (int u = 0; u < nr; u++) {
			int user = wusers.get(u);
			String userstring =rateDao.getUserId(user);
			String toprint = "";
			for (int j = 0; j < nc; j++) {
				toprint = toprint + IE.get(u, j) + " ";
			}
			toprint = userstring + " " + toprint.trim();
			System.out.println("IEMatrix " + toprint);
		}
		
		b = "";
		for (int j = 0; j < nc; j++) {
			b = b + IB.get(j) + " ";
		}
		System.out.println("IBVector " + b.trim());
		
		System.out.println("Compute maxE and minE");
		for (int u = 0; u < numUsers; u++) {
			if (u % 1000 == 0) {
				double percent = (double) u * (double) 100/(double) numUsers;
				System.out.println("Compute maxE and minE " + percent + "%");
			}
			for (Map.Entry<Integer, Integer> fi : testset.entrySet()) {
				int j = fi.getValue();
				double val = DenseMatrix.rowMult(E, u, itemContexts, j)  + B.inner(itemContexts.row(j));
				if (maxE < val) maxE = val;
				if (minE > val) minE = val;
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
				//currentscores.put(u, val);
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
					if (maxI < imageval) maxI = imageval;
					if (minI > imageval) minI = imageval;
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
	
	protected double predictT(int u, int j) {
		return DenseMatrix.rowMult(E, u, itemContexts, j) + B.inner(itemContexts.row(j));
	}
	
	protected double predictP(int u, int j) throws Exception {
		return DenseMatrix.rowMult(P, u, Q, j);
	}
	
	protected double predictTP(int u, int j) throws Exception {
		return DenseMatrix.rowMult(TP, u, TQ, j);
	}
	
	protected double predict(int u, int j) {
		if (Wscores.get(j)!=null) {
			if (Wscores.get(j).get(u)!=null) {
				return Wscores.get(j).get(u);
			}
		}

		if (Tscores.get(j)!=null) {
			if (Tscores.get(j).get(u)!=null) {
				return Tscores.get(j).get(u);
			}
		}
		
		double embeddingval = 0;
		if (Escores.get(j) != null) {
			if (Escores.get(j).get(u)!=null) {
				embeddingval = (Escores.get(j).get(u) - minE) / (maxE - minE);
			}
		}
		
		double imageval = 0;
		if (Iscores.get(j) != null) {
			if (Iscores.get(j).get(u)!=null) {
				imageval = (Iscores.get(j).get(u) - minI)/(maxI-minI);
			}
		}

		double val =  0;
		if (embeddingval > 0) {
			if (imageval > 0) {
				val = embeddingval + imageval;
			} else {
				val = embeddingval;
			}
		} else {
			if (imageval > 0) {
				val = imageval;
			} else {
				val = 0;
			}
		}
		
		return val;
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regC, numIters });
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
