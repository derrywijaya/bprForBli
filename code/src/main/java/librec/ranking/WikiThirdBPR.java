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
public class WikiThirdBPR extends ContextRecommender {
	private Map<Integer, Map<Integer, Double>> weights = new HashMap<Integer, Map<Integer, Double>>();
	private String lang;
	private Map<String, Integer> englishitemstring = new HashMap<String, Integer>();
	private Map<String, Integer> foreignitemstring = new HashMap<String, Integer>();
	private ArrayList<Integer> wikiusers = new ArrayList<Integer>();
	private ArrayList<Integer> wikiitems = new ArrayList<Integer>();
	private ArrayList<Integer> thirdusers = new ArrayList<Integer>();
	private ArrayList<Integer> thirditems = new ArrayList<Integer>();
	private Map<Integer, Integer> countthirditems = new HashMap<Integer, Integer>();
	private Map<Integer, Integer> foreignitem = new HashMap<Integer, Integer>();
	private DenseMatrix E, TP, TQ;
	private DenseVector B;
	private Map<Integer, Integer> trainitem = new HashMap<Integer, Integer>();
	
	public WikiThirdBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		initByNorm = false;
	}

	@Override
	protected void initModel() throws Exception {
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
				foreignitemstring.put(itemstring, f);
			}
		}
		lang = cf.getString("mainlanguage");
		if (lang.equalsIgnoreCase("eb")) lang = "ceb";
	}

	@Override
	protected void buildModel() throws Exception {
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
	}
	
	protected double predictP(int u, int j) throws Exception {
		return DenseMatrix.rowMult(P, u, Q, j);
	}
	
	protected double predictTP(int u, int j) throws Exception {
		return DenseMatrix.rowMult(TP, u, TQ, j);
	}
	
	protected double predict(int u, int j) {
		if (wikiusers.contains(u) && wikiitems.contains(j)) {
			u = wikiusers.indexOf(u);
			j = wikiitems.indexOf(j);
			double val2 = DenseMatrix.rowMult(P, u, Q, j);
			return val2;
		}

		if (thirdusers.contains(u) && thirditems.contains(j) && countthirditems.get(j) > 1) {
			u = thirdusers.indexOf(u);
			j = thirditems.indexOf(j);
			double val2 = DenseMatrix.rowMult(TP, u, TQ, j);
			return val2;
		}
		
		return 0;
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, regB, regC, numIters });
	}
}
