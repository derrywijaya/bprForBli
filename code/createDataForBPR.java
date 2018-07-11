import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * 
 * @author dwijaya
 * Assemble the input files for BPR to be in the required format from the supplied user embeddings
 *
 */

class createDataForBPR {
	public static void main(String args[]) {
		String foreign_words = args[0].trim();
		String foreign_embedding = args[1].trim();
		String language = args[2].trim();
		//String foreign_top100K = args[3].trim();
		String output_train = args[3].trim();
		String output_english_vector = args[4].trim();
		String output_foreign_vector = args[5].trim();
		String output_foreign_totranslate = args[6].trim();
		
		//Map<String,String> fwordsTop100K = new HashMap<String,String>();
		Map<String,String> evectors = new HashMap<String,String>();
		Map<String,String> fwordsextended = new HashMap<String,String>();
		Map<String,String> fvectors = new HashMap<String,String>();
		Map<String,String> train = new HashMap<String,String>();

		try {
			BufferedReader bfr = new BufferedReader(new FileReader(foreign_words));	
			String l;
			while ((l = bfr.readLine()) != null) {
				String t = l.toLowerCase().trim();
				if (t.length()>0) {
					t = "column-" + t.replace(",", "_");
					fwordsextended.put(t,t);
				}
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			BufferedReader bfr = new BufferedReader(new FileReader("context-"+language+"-en.txt"));	
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.toLowerCase().trim().split(" ");
				String word = temp[0].trim();
				l = "hellohello" + l.trim();
				String rest = l.replace("hellohello"+word, "").trim();
				evectors.put(word, rest);
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			BufferedReader bfr = new BufferedReader(new FileReader("context-"+language+"-"+language+".txt"));	
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.toLowerCase().trim().split(" ");
				String word = temp[0].trim();
				l = "hellohello" + l.trim();
				String rest = l.replace("hellohello"+word, "").trim();
				fvectors.put(word, rest);
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		/*try {
			BufferedReader bfr = new BufferedReader(new FileReader(foreign_top100K));	
			String l;
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.toLowerCase().trim().split(" ");
				String word = temp[0].trim();
				fwordsTop100K.put(word, word);
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}*/
		
		try {
			BufferedReader bfr = new BufferedReader(new FileReader(foreign_embedding));	
			String l;
			bfr.readLine();
			while ((l = bfr.readLine()) != null) {
				String temp[] = l.toLowerCase().trim().split(" ");
				String word = temp[0].trim();
				l = "hellohello" + l.trim();
				String rest = l.replace("hellohello"+word, "").trim();
				word = "column-" + word.replace(",", "_");
				//if ((fwordsextended.get(word)!=null || fwordsTop100K.get(word)!=null) && fvectors.get(word)==null) {
				if (fwordsextended.get(word)!=null && fvectors.get(word)==null) {
					fvectors.put(word,rest);
				}
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			BufferedReader bfr = new BufferedReader(new FileReader("train-"+language+".txt"));	
			String l;
			while ((l = bfr.readLine()) != null) {
				train.put(l.trim(), l.trim());
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			BufferedReader bfr = new BufferedReader(new FileReader("test-"+language+".txt"));	
			String l;
			while ((l = bfr.readLine()) != null) {
				train.put(l.trim(), l.trim());
			}
			bfr.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		for (Map.Entry<String, String> e : evectors.entrySet()) {
			String word = e.getKey();
			String s = word + " " + word + " 1";
			train.put(s,s);
		}
		
		try(FileWriter fw = new FileWriter(output_train, true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : train.entrySet()) {
				out.println(e.getKey());
			}
			out.close();
		} catch (IOException ex) {
		}
		
		try(FileWriter fw = new FileWriter(output_english_vector, true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : evectors.entrySet()) {
				out.println(e.getKey() + " " + e.getValue());
			}
			out.close();
		} catch (IOException ex) {
		}
		
		try(FileWriter fw = new FileWriter(output_foreign_vector, true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : fvectors.entrySet()) {
				out.println(e.getKey() + " " + e.getValue());
			}
			out.close();
		} catch (IOException ex) {
		}
		

		try(FileWriter fw = new FileWriter(output_foreign_totranslate, true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : fwordsextended.entrySet()) {
				out.println(e.getKey());
			}
			out.close();
		} catch (IOException ex) {
		}
	}
}