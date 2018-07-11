package librec.intf;

import java.util.Map;

import com.google.common.collect.Table;

import happy.coding.io.Logs;
import happy.coding.io.Strings;
import librec.data.DataDAO;
import librec.data.DenseMatrix;
//import librec.data.ItemContext;
import librec.data.RatingContext;
import librec.data.SparseMatrix;
//import librec.data.UserContext;

/**
 * Generic recommenders where contextual information is used. The context can be user-, item- and rating-related.
 * 
 * @author guoguibing
 * 
 */
public class ContextRecommender extends IterativeRecommender {
	protected static DataDAO contextDao;
	protected static DataDAO socialDao;
	// {user, user-context}
	//protected static Map<Integer, UserContext> userContexts;
	protected static DenseMatrix userContexts;

	protected static DenseMatrix itemContexts;
	protected static SparseMatrix socialMatrix;
	// {item, item-context}
	//protected static Map<Integer, ItemContext> itemContexts;
	
	// {user, item, rating-context}
	protected static Table<Integer, Integer, RatingContext> ratingContexts;

	// context regularization
	protected static float regC;

	// indicator of static field initialization or reset
	public static boolean resetStatics = true;

	protected static int numItemContexts;
	protected static int numUserContexts;
	
	// initialization
	static {

		// read context information here
		String contextPath = cf.getPath("dataset.context");
		Logs.debug("Context dataset: {}", Strings.last(contextPath, 38));

		contextDao = new DataDAO(contextPath, rateDao.getUserIds(), rateDao.getItemIds());

		try {
			System.out.println("HELLOFIRST\t" + rateDao.numItems() + "\t" + rateDao.numUsers() + "\t" + contextDao.numItems() + "\t" + contextDao.numUsers());
			itemContexts = contextDao.readDataDense();
			numItems = contextDao.numItems();
			numItemContexts = itemContexts.numColumns();
			System.out.println("HELLOSECOND\t" + rateDao.numItems() + "\t" + rateDao.numUsers() + "\t" + contextDao.numItems() + "\t" + contextDao.numUsers() + 
					"\t" + itemContexts.numColumns());
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		/*String socialPath = cf.getPath("dataset.social");
		Logs.debug("Social dataset: {}", Strings.last(contextPath, 38));

		socialDao = new DataDAO(socialPath, rateDao.getItemIds());

		try {
			socialMatrix = socialDao.readData();
			numItems = socialDao.numUsers();

			//socialCache = socialMatrix.rowCache(cacheSpec);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}*/

		/*try {
			userContexts = contextDao.readDataDenseT();
			numUsers = contextDao.numUsers();
			numUserContexts = userContexts.numColumns();
			
			//socialCache = socialMatrix.rowCache(cacheSpec);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}*/
	}

	public ContextRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		
		if (resetStatics) {
			resetStatics = false;
			regC = regOptions.getFloat("-c", reg);
		}
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { numFactors, initLRate, maxLRate, regB, regU, regI, regC, numIters,
				isBoldDriver });
	}
	
}
