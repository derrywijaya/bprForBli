import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * 
 * @author dwijaya
 * The second and final step of assembling input files for BPR to be in the required format 
 * from the supplied user embeddings
 */

class createDataForBPRFinal {
	public static void main(String args[]) {
		String language = args[0].trim();
		
		/*Map<String,String> top = new HashMap<String,String>();
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader("wiki."+language+".top.words"));	// read top 100K words
			String l1;
			while ((l1 = bfr1.readLine()) != null) {
				top.put(l1.trim(),l1.trim());
			}
			bfr1.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}*/
		
		Map<String,String> totranslate = new HashMap<String,String>();
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader(language+"-en.words"));	// read foreign words to translate
			String l1;
			while ((l1 = bfr1.readLine()) != null) {
				totranslate.put(l1.trim(),l1.trim());
			}
			bfr1.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Map<String,String> english = new HashMap<String,String>();
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader(language+"-en.vector.en.words"));	// read english words that have vectors
			String l1;
			while ((l1 = bfr1.readLine()) != null) {
				english.put(l1.trim(),l1.trim());
			}
			bfr1.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Map<String,String> foreign = new HashMap<String,String>();
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader(language+"-en.vector.extended.words"));	// read foreign words
			BufferedReader bfr2 = new BufferedReader(new FileReader(language+"-en.vector.extended.vector"));	// read foreign vectors
			String l1,l2;
			while ((l1 = bfr1.readLine()) != null) {
				l2 = bfr2.readLine();
				l1 = l1.trim();
				l2 = l2.trim();
				foreign.put(l1, l2);
			}
			bfr1.close();
			bfr2.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Map<String,String> traintowrite = new HashMap<String,String>();
		Map<String,String> dictionary = new HashMap<String,String>();
		Map<String,ArrayList<String>> alldictionary = new HashMap<String,ArrayList<String>>();
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader(language + "-dict-mturk.txt"));	// read dictionary
			String l1;
			while ((l1 = bfr1.readLine()) != null) {
				String temp[] = l1.trim().split(" ");
				String en = temp[0].trim();
				String fo = temp[1].trim();
				ArrayList<String> al = alldictionary.get(fo);
				if (al==null) al = new ArrayList<String>();
				if (!al.contains(en))al.add(en);
				alldictionary.put(fo, al);
				if (english.get(en)!=null && foreign.get(fo)!=null) {
					String s = en + " " + en + " 1";
					traintowrite.put(s, s);
					traintowrite.put(l1.trim(), l1.trim());
					dictionary.put(fo, fo);
				}
			}
			bfr1.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (Map.Entry<String, String> e : english.entrySet()) {
			String en = e.getKey();
			String s = en + " " + en + " 1";
			traintowrite.put(s, s);
		}
		
		File file = new File(language + ".train");
		file.delete();
		try(FileWriter fw = new FileWriter(language + ".train", true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : traintowrite.entrySet()) {
				out.println(e.getKey());
			}
			out.close();
		} catch (IOException ex) {
		}
		
		file = new File(language + ".train.vec");
		file.delete();
		try(FileWriter fw = new FileWriter(language + ".train.vec", true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : foreign.entrySet()) {
				String fo = e.getKey();
				String ve = e.getValue();
				if (dictionary.get(fo)!=null) {
					out.println(fo + " " + ve);					
				}
			}
			out.close();
		} catch (IOException ex) {
		}
		
		file = new File(language + ".translated");
		file.delete();
		try(FileWriter fw = new FileWriter(language + ".translated", true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : totranslate.entrySet()) {
				String fo = e.getKey();
				if (alldictionary.get(fo)!=null) {
					ArrayList<String> al = alldictionary.get(fo);
					String s = "";
					for (String a : al) {
						s = s + (s.length()>0 ? "\t" : "") + a; 
					}
					out.println(fo + "\t" + s.trim());		
				}
			}
			out.close();
		} catch (IOException ex) {
		}
		
		file = new File(language + ".totranslate");
		file.delete();
		try(FileWriter fw = new FileWriter(language + ".totranslate", true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : totranslate.entrySet()) {
				String fo = e.getKey();
				if (foreign.get(fo)!=null) {
					out.println(fo);		
				}
			}
			out.close();
		} catch (IOException ex) {
		}
		
		file = new File(language + ".extended");
		file.delete();
		try(FileWriter fw = new FileWriter(language + ".extended", true);
				BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw))
		{
			for (Map.Entry<String, String> e : foreign.entrySet()) {
				String fo = e.getKey();
				String ve = e.getValue();
				//if (top.get(fo)!=null || totranslate.get(fo)!=null || dictionary.get(fo)!=null) {
				if (totranslate.get(fo)!=null || dictionary.get(fo)!=null) {
					out.println(fo + " " + ve);					
				}
			}
			out.close();
		} catch (IOException ex) {
		}
	}
}