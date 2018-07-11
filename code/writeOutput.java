import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

class writeOutput {
	public static void main(String args[]) {
		Map<String,ArrayList<String>> additional = new HashMap<String,ArrayList<String>>();
		Map<String,ArrayList<String>> additionalscores = new HashMap<String,ArrayList<String>>();
		Map<String,String> written = new HashMap<String,String>();
		String predictionfile = args[0].trim();
		String translatedfile = args[1].trim();
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader(predictionfile));	// read prediction file
			String l1;
			bfr1.readLine();
			while ((l1 = bfr1.readLine()) != null) {
				String line = l1.trim();
				String temp[] = line.split(": \\(");
				String temp1[] = temp[0].trim().split(": column\\-");
				String foreign = temp1[1].trim();
				String trans = temp[1].trim();
				trans = trans.substring(0, trans.length()-1);
				String ens[] = trans.split("\\), \\(");
				ArrayList<String> adds = new ArrayList<String>();
				ArrayList<String> scores = new ArrayList<String>();
				for (int i = 0; i < ens.length; i++) {
					String tran[] = ens[i].trim().split(", ");
					String en = tran[0].replaceFirst("row-", "").trim();
					String score = tran[1].trim();
					if (score.equalsIgnoreCase("Infinity")) score = "2.0";
					adds.add(en);
					scores.add(score);
				}
				additional.put(foreign, adds);
				additionalscores.put(foreign, scores);
			}
			bfr1.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader(translatedfile));	// read translated file
			String l1;
			while ((l1 = bfr1.readLine()) != null) {
				String temp[] = l1.split("\t");
				String foreign = temp[0].trim();
				foreign = foreign.replaceFirst("column-", "").trim();
				written.put(foreign, foreign);
				String s = foreign + "\t";
				Map<String,String> alreadywritten = new HashMap<String,String>();
				for (int i = 1; i < temp.length; i++) {
					String english = temp[i].replaceFirst("row-", "").trim();
					s = s + english + " (2.0), ";
					alreadywritten.put(english, english);
				}
				if (additional.get(foreign)!=null) {
					ArrayList<String> adds = additional.get(foreign);
					ArrayList<String> scores = additionalscores.get(foreign);
					for (int j = 0; j < adds.size(); j++) {
						String add = adds.get(j);
						String score = scores.get(j);
						if (alreadywritten.get(add)==null) {
							s = s + add + " (" + score + "), ";
						}
					}
				}
				s = s.substring(0, s.length()-2);
				System.out.println(s);
			}
			bfr1.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			BufferedReader bfr1 = new BufferedReader(new FileReader(predictionfile));	// print out the rest of the prediction
			String l1;
			bfr1.readLine();
			while ((l1 = bfr1.readLine()) != null) {
				String line = l1.trim();
				String temp[] = line.split(": \\(");
				String temp1[] = temp[0].trim().split(": column\\-");
				String foreign = temp1[1].trim();
				if (written.get(foreign)!=null) continue; // already written above
				String trans = temp[1].trim();
				trans = trans.substring(0, trans.length()-1);
				String ens[] = trans.split("\\), \\(");
				String s = foreign + "\t";
				for (int i = 0; i < ens.length; i++) {
					String tran[] = ens[i].trim().split(", ");
					String en = tran[0].replaceFirst("row-", "").trim();
					String score = tran[1].trim();
					if (score.equalsIgnoreCase("Infinity")) score = "2.0";
					s = s + en + " (" + score + "), ";
				}
				s = s.substring(0, s.length()-2);
				System.out.println(s);
			}
			bfr1.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}