import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * 
 * @author dwijaya
 * Assemble the train and development input embeddings for learning the mapping 
 * between the embedding spaces
 * 
 */

class createTrainAndDevMapping {
	static String foreignwordfile, foreignvectorfile, englishwordfile, 
		englishvectorfile, seedfile, name;
	static Map<String,String> foreignvectors = new HashMap<String,String>();
	static Map<String,String> englishvectors = new HashMap<String,String>();
	static Map<String, String> englishToForeign = new HashMap<String,String>();
	
	public static void main(String args[]) {
		foreignwordfile = args[0].trim();
		foreignvectorfile = args[1].trim();
		englishwordfile = args[2].trim();
		englishvectorfile = args[3].trim();
		seedfile = args[4].trim();
		name = args[5].trim();
		
		System.out.println("read seed file");
		Map<String,String> ens = new HashMap<String,String>();
		Map<String,String> fls = new HashMap<String,String>();
		try {
			BufferedReader bfr = getFileReader(seedfile);	
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.split(" ");
				String en = temp[0].trim();
				String fl = temp[1].trim();
				ens.put(en, en);
				fls.put(fl, fl);
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("read foreign file");

		try {
			BufferedReader bfrw = getFileReader(foreignwordfile);	
			BufferedReader bfrv = getFileReader(foreignvectorfile);	
			String lw,lv;
			while ((lw = bfrw.readLine()) != null) {
				lv = bfrv.readLine();
				lw = lw.trim();
				lv = lv.trim();
				if (fls.get(lw)!=null && lv.length() > 0) {
					foreignvectors.put(lw, lv);					
				}
			}
			bfrw.close();
			bfrv.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("read english file");

		try {
			BufferedReader bfrw = getFileReader(englishwordfile);	
			BufferedReader bfrv = getFileReader(englishvectorfile);	
			String lw,lv;
			while ((lw = bfrw.readLine()) != null) {
				lv = bfrv.readLine();
				lw = lw.trim();
				lv = lv.trim();
				if (ens.get(lw)!=null && lv.length() > 0) {
					englishvectors.put(lw, lv);					
				}
			}
			bfrw.close();
			bfrv.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("read seed file again");

		try {
			BufferedReader bfr = getFileReader(seedfile);	
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.split(" ");
				String en = temp[0].trim();
				String fl = temp[1].trim();
				if (englishvectors.get(en)!=null && foreignvectors.get(fl)!=null) {
					englishToForeign.put(en + "\t" + fl, en + "\t" + fl);
				}
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("start to write");

		int total = englishToForeign.size();
		int testsize = (int) Math.ceil((double) total * 0.1);
		int trainsize = total - testsize;
		System.out.println(total + " " + foreignvectors.size() + " " + englishvectors.size());
		int idx = 0;
		boolean appendinput = false;
		boolean appendoutput = false;
		boolean appendtestinput = false;
		boolean appendtestoutput = false;
		for (Map.Entry<String, String> entry : englishToForeign.entrySet()) {
			String text = entry.getKey();
			String e = text.split("\t")[0].trim();
			String erest = englishvectors.get(e);
			String f = text.split("\t")[1].trim();
			String frest = foreignvectors.get(f);
			if (idx < trainsize) {
				writeToFile(erest, name+"-input-train.txt", appendinput);
				appendinput = true;
				writeToFile(frest, name+"-output-train.txt", appendoutput);
				appendoutput = true;
			} else {
				writeToFile(erest, name+"-input-test.txt", appendtestinput);
				appendtestinput = true;
				writeToFile(frest, name+"-output-test.txt", appendtestoutput);
				appendtestoutput = true;
			}
			idx++;
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
	 
	public static BufferedReader getFileReader(String fileName) throws IOException {
		  if (fileName.endsWith("gz")) {
		    return new BufferedReader(new InputStreamReader(
		        new GZIPInputStream(new FileInputStream(fileName))));
		  } else {
		    return new BufferedReader(new FileReader(fileName));
		  }
	}
}