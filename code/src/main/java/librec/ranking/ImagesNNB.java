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
import librec.data.DataDAO;
import librec.data.DenseMatrix;
import librec.data.SparseMatrix;
import librec.intf.IterativeRecommender;

/**
 * Context-based Bayesian Personalized Ranking (CBPR)
 * 
 * @author guoguibing
 * 
 */
public class ImagesNNB extends IterativeRecommender {
	private String lang;
	private String mappingfile;
	private Map<String, Integer> fitemstring = new HashMap<String, Integer>();
	private Map<String, Integer> eitem = new HashMap<String, Integer>();
	private Map<Integer, Map<Integer, Double>> Iscores = new HashMap<Integer, Map<Integer, Double>>();
	private DenseMatrix imageContexts;
	private int numImages, numImageContexts;
	private Map<Integer, Map<Integer, Integer>> targetimagemapping = new HashMap<Integer, Map<Integer, Integer>>();
	private Map<Integer, Map<Integer, Integer>> sourceimagemapping = new HashMap<Integer, Map<Integer, Integer>>();
	private DataDAO imageDao;
	
	public ImagesNNB(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
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
		
		try {
			BufferedReader bfr = new BufferedReader(new FileReader(mappingfile));		
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.split("\t");
				String word = temp[0].trim();
				String imageword = temp[1].trim();
				if (eitem.get(word)!=null) {
					int user = eitem.get(word);
					Map<Integer, Integer> map = targetimagemapping.get(user);
					if (map == null) map = new HashMap<Integer, Integer>();
					int imageitem = imageDao.getItemId(imageword);
					map.put(imageitem, imageitem);
					targetimagemapping.put(user, map);	
				} else {
					if (fitemstring.get(word)!=null) {
						int item = fitemstring.get(word);
						Map<Integer, Integer> map = sourceimagemapping.get(item);
						if (map == null) map = new HashMap<Integer, Integer>();
						int imageitem = imageDao.getItemId(imageword);
						map.put(imageitem, imageitem);
						sourceimagemapping.put(item, map);	
					}
				}
			} 
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("Compute maxI and minI");
		for (int u = 0; u < numUsers; u++) {
			if (u % 1000 == 0) {
				double percent = (double) u * (double) 100/(double) numUsers;
				System.out.println("Compute maxI and minI " + percent + "%");
			}
			for (Map.Entry<Integer, Integer> fi : testset.entrySet()) {
				int j = fi.getValue();
				if (sourceimagemapping.get(j)!=null && targetimagemapping.get(u)!=null) {
					double imageval = 0;
					Map<Integer, Integer> imageusers = targetimagemapping.get(u);
					Map<Integer, Integer> imageitems = sourceimagemapping.get(j);
					for (Map.Entry<Integer, Integer> e2 : imageitems.entrySet()) {
						int imageitem = e2.getKey();
						double max = -Double.MAX_VALUE;
						for (Map.Entry<Integer, Integer> e3 : imageusers.entrySet()) {
							int useritem = e3.getKey();
							double val = DenseMatrix.rowMult(imageContexts, useritem, imageContexts, imageitem);
							if (val>max) max = val; 
						}
						imageval = imageval + max;
					}
					imageval = imageval / (double) imageitems.size();
					Map<Integer, Double> currentscores = Iscores.get(j);
					if (currentscores == null) currentscores = new HashMap<Integer, Double>();
					if (currentscores.size() < 10) {
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
					Iscores.put(j, currentscores);
				}
			}
		}
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
