import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPInputStream;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * @author dwijaya
 * Read bilingual dictionary in JSON format and assemble test and train sets 
 * for training and testing BPR 
 * 
 */

class readDictionary {
	public static void main(String args[]) {
		String json_dictionary = args[0].trim(); // bilingual dictionary
		String english_embedding = args[1].trim(); // english embeddings
		String foreign_embedding = args[2].trim(); // foreign embeddings
		String output_train = args[3].trim(); // output train file
		String output_test = args[4].trim(); // output test file
		
		JSONParser parser = new JSONParser();
		Map<String, Map<String, String>> origtrain = new HashMap<String, Map<String, String>>();
		readData(parser, json_dictionary, origtrain);
		
		Map<String,String> ews = new HashMap<String,String>();
		try {
			BufferedReader bfr = getFileReader("wiki.en.top.words");			
			String l;
			while ((l = bfr.readLine()) != null) {
				String word = l.toLowerCase().trim();
				ews.put(word, word);
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Map<String, String> features = new HashMap<String,String>();
		Map<String, String> writtentest = new HashMap<String, String>();
		Map<String, String> writtenwords = new HashMap<String, String>();
		Map<String, String> traindata = new HashMap<String,String>();
		Map<String, String> testdata = new HashMap<String,String>();
		try {
			BufferedReader bfr = getFileReader(english_embedding);			
			String l;
			while ((l = bfr.readLine()) != null) {
				l = l.toLowerCase().trim();			
				String temp[] = l.split(" ");
				String word = temp[0].trim();
				if (ews.get(word)!=null) {
					String t = word + " " + word + " 1";
					traindata.put(t, t);
					writtenwords.put(word, word);
					System.out.println(l);
				}
				features.put(word, l);
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Map<String,String> candidatetest = new HashMap<String, String>();
		ArrayList<String> candidatetestarrayorig = new ArrayList<String>();
		ArrayList<String> candidatetestarray = new ArrayList<String>();
		for (Map.Entry<String, Map<String, String>> e : origtrain.entrySet()) {
			String foreign = e.getKey().trim();
			Map<String, String> vals = e.getValue();
			String rf = "column-" + foreign.replace(" ", "_").replace(",", "_");
			ArrayList<String> candidates = new ArrayList<String>();
			ArrayList<String> candidatesorig = new ArrayList<String>();
			for (Map.Entry<String, String> e1 : vals.entrySet()) {
				String ew = e1.getKey().trim();
				String ce = "row-" + ew.replace(" ", "_").replace(",", "_");
				if (features.get(ce)!=null) {
					candidates.add(ce);
					candidatesorig.add(ew);
				}
			}
			if (candidates.size()==1) {
				String chosenword = candidates.get(0);
				candidatetest.put(rf, chosenword);
				candidatetestarrayorig.add(foreign);
				candidatetestarray.add(rf);
			}
		}
		
		Map<String,String> test = new HashMap<String, String>();
		int size = candidatetestarray.size();
		if (size > 1000) {
			Random r = new Random();
			Map<Integer, Integer> alreadychosen = new HashMap<Integer, Integer>();
			while (test.size() < 1000) {
				int chosen = r.nextInt(size);
				if (alreadychosen.get(chosen)==null) {
					alreadychosen.put(chosen, chosen);
					String rf = candidatetestarray.get(chosen);
					String foreign = candidatetestarrayorig.get(chosen);
					test.put(rf, candidatetest.get(rf));
					origtrain.remove(foreign);
				}
			}
		} else {
			for (int i = 0; i < size; i++) {
				String rf = candidatetestarray.get(i);
				String foreign = candidatetestarrayorig.get(i);
				test.put(rf, candidatetest.get(rf));
				origtrain.remove(foreign);
			}
		}

		try {
			BufferedReader bfr = getFileReader(foreign_embedding);			
			String l;
			while ((l = bfr.readLine()) != null) {
				l = l.toLowerCase().trim();
				String temp[] = l.split(" ");
				String word = temp[0].trim();
				if (test.get(word)!=null) {
					String row = test.get(word);
					if (features.get(row)!=null) {
						String s = row + " " + word + " 1";
						testdata.put(s, s);
						writtentest.put(s, s);
						System.out.println(l);
						writtenwords.put(word, word);	
						if (writtenwords.get(row)==null) {
							String t = row + " " + row + " 1";
							traindata.put(t, t);
							writtenwords.put(row, row);
							System.out.println(features.get(row));
						}
					}
				}
				features.put(word, l);
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (Map.Entry<String, Map<String, String>> e : origtrain.entrySet()) {
			String foreign = e.getKey().trim();
			Map<String, String> vals = e.getValue();
			String rf = "column-" + foreign.replace(" ", "_").replace(",", "_");
			if (features.get(rf)==null) continue;
			for (Map.Entry<String, String> e1 : vals.entrySet()) {
				String ew = e1.getKey().trim();
				String ce = "row-" + ew.replace(" ", "_").replace(",", "_");
				if (features.get(ce)==null) continue;
				String t = ce + " " + rf + " 1";
				if (writtentest.get(t)==null) {
					traindata.put(t, t);
					if (writtenwords.get(rf)==null) {
						writtenwords.put(rf, rf);
						System.out.println(features.get(rf));
					}
					if (writtenwords.get(ce)==null) {
						writtenwords.put(ce, ce);
						System.out.println(features.get(ce));
					}
				}
			}
		}
		
		boolean append = false;
		for (Map.Entry<String, String> e : traindata.entrySet()) {
			writeToFile(e.getKey(), output_train, append);
			append = true;
		}
		
		append = false;
		for (Map.Entry<String, String> e : testdata.entrySet()) {
			writeToFile(e.getKey(), output_test, append);
			append = true;
		}
		
	}

	public static BufferedReader getFileReader(String fileName) throws IOException {
		  if (fileName.endsWith("gz")) {
		    return new BufferedReader(new InputStreamReader(
		        new GZIPInputStream(new FileInputStream(fileName))));
		  } else {
		    return new BufferedReader(new FileReader(fileName));
		  }
	}
	 
	 public static void writeToFile(String toWrite, String filename, boolean append) {
			try {
				FileWriter fstream = new FileWriter(filename, append);
				BufferedWriter out = new BufferedWriter(fstream);
				out.write(toWrite+"\n");
				out.close();
			} catch(Exception e) {
				System.out.println("Error: " + e.getMessage());
			}
		}
	
	@SuppressWarnings("unchecked")
	private static void readData(JSONParser parser, String trainFile, Map<String, Map<String, String>> train) {
		 try {
				BufferedReader bfr = getFileReader(trainFile);			
				String l;
				while ((l = bfr.readLine()) != null) {
					Object obj = parser.parse(l);
					JSONObject jsonObject = (JSONObject) obj;
					for (Object e : jsonObject.keySet()) {
						String key = (String) e;
	 					JSONArray values = (JSONArray) jsonObject.get(key);
						Iterator<String> iterator = values.iterator();
						Map<String, String> translations = new HashMap<String, String>();
						while (iterator.hasNext()) {
							String translation[] = iterator.next().toLowerCase().split("[\\,\\(\\)\\;\\/\\=\\:]");
							for (String t : translation) {
								t = t.trim();
								int length = t.split(" ").length;
								if (t.length() > 0 && length <4) {
									translations.put(t, t);
								}
							}
						}
						if (translations.size() > 0) {
							key = key.toLowerCase();
							Map<String, String> current = train.get(key);
							if (current == null) current = new HashMap<String, String>();
							for (Map.Entry<String, String> entry: translations.entrySet()) {
								String t = entry.getKey();
								if (!t.equalsIgnoreCase(key)) {
									current.put(t, t);								
								}
							}
							if (current.size()>0) train.put(key, current);
						}
					}
				}
				bfr.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ParseException e) {
				e.printStackTrace();
			}
	}

}