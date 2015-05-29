package ml.recommendation.util;

import java.util.*;

import la.io.*;
import la.vector.*;
import la.vector.Vector;

import java.io.*;

import ml.utils.*;

public class Utility
{
    public static boolean drawMAPCurve;
    
    static {
        Utility.drawMAPCurve = false;
    }
    
    public static void exit(final int status) {
        System.exit(status);
    }
    
    public static <K, V> void saveMap(final Map<K, V> map, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            e.printStackTrace();
            exit(1);
        }
        for (final Map.Entry<K, V> entry : map.entrySet()) {
            pw.print(entry.getKey());
            pw.print('\t');
            pw.println(entry.getValue());
        }
        pw.close();
    }
    
    public static void addTreeStructuredGroupList(final Node parent, final ArrayList<ArrayList<Pair<Integer, Integer>>> treeStructuredPairGroupList) {
        if (parent.children == null) {
            final ArrayList<Pair<Integer, Integer>> pairList = nodeEdges(parent);
            if (pairList != null) {
                treeStructuredPairGroupList.add(pairList);
            }
            return;
        }
        final TreeMap<Integer, Node> children = parent.children;
        for (final int idx : children.keySet()) {
            final Node child = children.get(idx);
            addTreeStructuredGroupList(child, treeStructuredPairGroupList);
        }
        final ArrayList<Pair<Integer, Integer>> pairList2 = nodeEdges(parent);
        if (pairList2 != null) {
            treeStructuredPairGroupList.add(pairList2);
        }
    }
    
    private static ArrayList<Pair<Integer, Integer>> nodeEdges0(final Node parent) {
        if (parent.idx == 0 || parent.parentIdx == 0) {
            return null;
        }
        if (parent.children == null) {
            final ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
            pairList.add(Pair.of(parent.parentIdx, parent.idx));
            return pairList;
        }
        final ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
        final TreeMap<Integer, Node> children = parent.children;
        for (final int idx : children.keySet()) {
            final Node child = children.get(idx);
            pairList.addAll(nodeEdges0(child));
        }
        pairList.add(Pair.of(parent.parentIdx, parent.idx));
        return pairList;
    }
    
    private static ArrayList<Pair<Integer, Integer>> nodeEdges(final Node parent) {
        if (parent.idx == 0 || parent.parentIdx == 0) {
            return null;
        }
        if (parent.children == null) {
            final ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
            pairList.add(Pair.of(parent.parentIdx, parent.idx));
            return pairList;
        }
        final ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
        final TreeMap<Integer, Node> children = parent.children;
        for (final int idx : children.keySet()) {
            final Node child = children.get(idx);
            pairList.addAll(nodeEdges(child));
        }
        pairList.add(0, Pair.of(parent.parentIdx, parent.idx));
        return pairList;
    }
    
    public static void traverseTree(final Node parent, final HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, final TreeMap<Integer, Pair<Integer, Integer>> index2PairMap) {
        if (parent.children == null) {
            if (parent.idx == 0 || parent.parentIdx == 0) {
                return;
            }
            final Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
            final int index = pair2IndexMap.size();
            pair2IndexMap.put(pair, index);
            index2PairMap.put(index, pair);
        }
        else {
            final TreeMap<Integer, Node> children = parent.children;
            for (final int idx : children.keySet()) {
                final Node child = children.get(idx);
                traverseTree(child, pair2IndexMap, index2PairMap);
            }
            if (parent.idx == 0 || parent.parentIdx == 0) {
                return;
            }
            final Pair<Integer, Integer> pair2 = Pair.of(parent.parentIdx, parent.idx);
            final int index2 = pair2IndexMap.size();
            pair2IndexMap.put(pair2, index2);
            index2PairMap.put(index2, pair2);
        }
    }
    
    public static void postTraverse(final Node parent, final HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, final TreeMap<Integer, Pair<Integer, Integer>> index2PairMap) {
        if (parent.children == null) {
            if (parent.idx == 0 || parent.parentIdx == 0) {
                return;
            }
            final Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
            final int index = pair2IndexMap.size();
            pair2IndexMap.put(pair, index);
            index2PairMap.put(index, pair);
        }
        else {
            final TreeMap<Integer, Node> children = parent.children;
            for (final int idx : children.keySet()) {
                final Node child = children.get(idx);
                postTraverse(child, pair2IndexMap, index2PairMap);
            }
            if (parent.idx == 0 || parent.parentIdx == 0) {
                return;
            }
            final Pair<Integer, Integer> pair2 = Pair.of(parent.parentIdx, parent.idx);
            final int index2 = pair2IndexMap.size();
            pair2IndexMap.put(pair2, index2);
            index2PairMap.put(index2, pair2);
        }
    }
    
    public static void preTraverse(final Node parent, final HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, final TreeMap<Integer, Pair<Integer, Integer>> index2PairMap) {
        if (parent.parentIdx != 0 && parent.idx != 0) {
            final Pair<Integer, Integer> pair = Pair.of(parent.parentIdx, parent.idx);
            final int index = pair2IndexMap.size();
            pair2IndexMap.put(pair, index);
            index2PairMap.put(index, pair);
        }
        final TreeMap<Integer, Node> children = parent.children;
        if (children == null) {
            return;
        }
        for (final int idx : children.keySet()) {
            final Node child = children.get(idx);
            preTraverse(child, pair2IndexMap, index2PairMap);
        }
    }
    
    public static void buildTreeStructuredIndexGroupList(final ArrayList<ArrayList<Pair<Integer, Integer>>> treeStructuredPairGroupList, final HashMap<Pair<Integer, Integer>, Integer> pair2IndexMap, final ArrayList<ArrayList<Integer>> treeStructuredIndexGroupList) {
        for (final ArrayList<Pair<Integer, Integer>> pairList : treeStructuredPairGroupList) {
            final ArrayList<Integer> indexList = new ArrayList<Integer>();
            for (final Pair<Integer, Integer> pair : pairList) {
                final int index = pair2IndexMap.get(pair);
                indexList.add(index);
            }
            treeStructuredIndexGroupList.add(indexList);
        }
    }
    
    public static <T> void saveTreeStructuredGroupList(final ArrayList<ArrayList<T>> treeStructuredGroupList, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        for (final ArrayList<T> list : treeStructuredGroupList) {
            final StringBuilder sb = new StringBuilder(10);
            for (final T element : list) {
                sb.append(element);
                sb.append("\t");
            }
            pw.println(sb.toString().trim());
        }
        pw.close();
    }
    
    public static int[] loadTestUserEventRelation(final String eventFilePath, final HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap) {
        if (!new File(eventFilePath).exists()) {
            System.err.println(String.format("Event file %s doesn't exist.\n", eventFilePath));
            exit(1);
        }
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(eventFilePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        String line = "";
        final List<Double> YijList = new LinkedList<Double>();
        final List<double[]> XijList = new LinkedList<double[]>();
        final List<Integer> userIdxList = new LinkedList<Integer>();
        final List<Integer> itemIdxList = new LinkedList<Integer>();
        String[] container = null;
        double label = 0.0;
        int userIdx = -1;
        int itemIdx = -1;
        double gmp = 0.0;
        double freshness = 0.0;
        int eventIdx = -1;
        int maxUserIdx = -1;
        int maxItemIdx = -1;
        try {
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                container = line.split("\t");
                label = Double.parseDouble(container[0]);
                userIdx = Integer.parseInt(container[1]);
                itemIdx = Integer.parseInt(container[2]);
                if (maxUserIdx < userIdx) {
                    maxUserIdx = userIdx;
                }
                if (maxItemIdx < itemIdx) {
                    maxItemIdx = itemIdx;
                }
                gmp = Double.parseDouble(container[3]);
                freshness = Double.parseDouble(container[4]);
                YijList.add(label);
                userIdxList.add(userIdx);
                itemIdxList.add(itemIdx);
                XijList.add(new double[] { gmp, freshness });
                ++eventIdx;
                if (TestUser2EventIndexSetMap.containsKey(userIdx)) {
                    TestUser2EventIndexSetMap.get(userIdx).add(eventIdx);
                }
                else {
                    final LinkedList<Integer> eventSet = new LinkedList<Integer>();
                    eventSet.add(eventIdx);
                    TestUser2EventIndexSetMap.put(userIdx, eventSet);
                }
            }
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        final int eventCnt = YijList.size();
        final int[] TestUserIndices = new int[eventCnt];
        int cnt = 0;
        cnt = 0;
        for (final int element : userIdxList) {
            TestUserIndices[cnt++] = element;
        }
        return TestUserIndices;
    }
    
    public static Data loadData(final String appDirPath, final String eventFileName, final String userFileName, final String itemFileName, final int[] featureSizes) {
        final Data data = new Data();
        final int Pu = featureSizes[0];
        final int Pv = featureSizes[1];
        final int Pe = featureSizes[2];
        data.Pu = Pu;
        data.Pv = Pv;
        data.Pe = Pe;
        System.out.println("Loading events...");
        int M = 0;
        int N = 0;
        int T = 0;
        double[] Yij = null;
        double[][] Xij = null;
        int[] UserIndices = null;
        int[] ItemIndices = null;
        final HashMap<Integer, LinkedList<Integer>> CUser = new HashMap<Integer, LinkedList<Integer>>();
        final HashMap<Integer, LinkedList<Integer>> CItem = new HashMap<Integer, LinkedList<Integer>>();
        final HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
        final HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
        final String eventFilePath = String.valueOf(appDirPath) + File.separator + eventFileName;
        if (!new File(eventFilePath).exists()) {
            System.err.println(String.format("Event file %s doesn't exist.\n", eventFilePath));
            exit(1);
        }
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(eventFilePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        String line = "";
        final List<Double> YijList = new LinkedList<Double>();
        final List<double[]> XijList = new LinkedList<double[]>();
        final List<Integer> userIdxList = new LinkedList<Integer>();
        final List<Integer> itemIdxList = new LinkedList<Integer>();
        String[] container = null;
        double label = 0.0;
        int userIdx = -1;
        int itemIdx = -1;
        double gmp = 0.0;
        double freshness = 0.0;
        int eventIdx = -1;
        try {
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                container = line.split("\t");
                label = Double.parseDouble(container[0]);
                userIdx = Integer.parseInt(container[1]);
                itemIdx = Integer.parseInt(container[2]);
                gmp = Double.parseDouble(container[3]);
                freshness = Double.parseDouble(container[4]);
                YijList.add(label);
                userIdxList.add(userIdx);
                itemIdxList.add(itemIdx);
                XijList.add(new double[] { gmp, freshness });
                if (CUser.containsKey(userIdx)) {
                    CUser.get(userIdx).add(itemIdx);
                }
                else {
                    final LinkedList<Integer> itemSet = new LinkedList<Integer>();
                    itemSet.add(itemIdx);
                    CUser.put(userIdx, itemSet);
                }
                if (CItem.containsKey(itemIdx)) {
                    CItem.get(itemIdx).add(userIdx);
                }
                else {
                    final LinkedList<Integer> userSet = new LinkedList<Integer>();
                    userSet.add(userIdx);
                    CItem.put(itemIdx, userSet);
                }
                ++eventIdx;
                if (User2EventIndexSetMap.containsKey(userIdx)) {
                    User2EventIndexSetMap.get(userIdx).add(eventIdx);
                }
                else {
                    final LinkedList<Integer> eventSet = new LinkedList<Integer>();
                    eventSet.add(eventIdx);
                    User2EventIndexSetMap.put(userIdx, eventSet);
                }
                if (Item2EventIndexSetMap.containsKey(itemIdx)) {
                    Item2EventIndexSetMap.get(itemIdx).add(eventIdx);
                }
                else {
                    final LinkedList<Integer> eventSet = new LinkedList<Integer>();
                    eventSet.add(eventIdx);
                    Item2EventIndexSetMap.put(itemIdx, eventSet);
                }
            }
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        final int eventCnt = YijList.size();
        Yij = new double[eventCnt];
        Xij = new double[eventCnt][];
        UserIndices = new int[eventCnt];
        ItemIndices = new int[eventCnt];
        int cnt = 0;
        cnt = 0;
        for (final double element : YijList) {
            Yij[cnt++] = element;
        }
        cnt = 0;
        for (final double[] element2 : XijList) {
            Xij[cnt++] = element2;
        }
        cnt = 0;
        for (final int element3 : userIdxList) {
            UserIndices[cnt++] = element3;
        }
        cnt = 0;
        for (final int element3 : itemIdxList) {
            ItemIndices[cnt++] = element3;
        }
        M = CUser.size();
        N = CItem.size();
        T = eventCnt;
        String filePath = "";
        DataVectors dataVectors = null;
        System.out.println("Loading users...");
        Vector[] Xi = null;
        filePath = String.valueOf(appDirPath) + File.separator + userFileName;
        DataVectors.IdxStart = 0;
        try {
            dataVectors = DataVectors.readDataSetFromFile(filePath);
        }
        catch (IOException e3) {
            e3.printStackTrace();
        }
        catch (InvalidInputDataException e4) {
            e4.printStackTrace();
        }
        Xi = dataVectors.Vs;
        System.out.println("Loading items...");
        Vector[] Xj = null;
        filePath = String.valueOf(appDirPath) + File.separator + itemFileName;
        DataVectors.IdxStart = 0;
        try {
            dataVectors = DataVectors.readDataSetFromFile(filePath);
        }
        catch (IOException e5) {
            e5.printStackTrace();
        }
        catch (InvalidInputDataException e6) {
            e6.printStackTrace();
        }
        Xj = dataVectors.Vs;
        for (int j = 0; j < N; ++j) {
            ((SparseVector)Xj[j]).setDim(Pv);
        }
        data.M = M;
        data.N = N;
        data.T = T;
        data.Yij = Yij;
        data.Xij = Xij;
        data.UserIndices = UserIndices;
        data.ItemIndices = ItemIndices;
        data.CUser = CUser;
        data.CItem = CItem;
        data.User2EventIndexSetMap = User2EventIndexSetMap;
        data.Item2EventIndexSetMap = Item2EventIndexSetMap;
        data.Xi = Xi;
        data.Xj = Xj;
        return data;
    }
    
    public static int[] getFeatureSize(final String featureSizeFilePath) {
        BufferedReader br = null;
        String line = "";
        String[] container = null;
        final String lastLine = "";
        int Pu = 0;
        int Pv = 0;
        int Pe = 0;
        if (!new File(featureSizeFilePath).exists()) {
            System.err.println(String.format("File %s doesn't exist.\n", featureSizeFilePath));
            exit(1);
        }
        try {
            br = new BufferedReader(new FileReader(featureSizeFilePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        try {
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                container = lastLine.split("\t");
                if (container[0].equals("User")) {
                    Pu = Integer.parseInt(container[1]);
                }
                else if (container[0].equals("Item")) {
                    Pv = Integer.parseInt(container[1]);
                }
                else {
                    if (!container[0].equals("Event")) {
                        continue;
                    }
                    Pe = Integer.parseInt(container[1]);
                }
            }
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        final int[] featureSizes = { Pu, Pv, Pe };
        return featureSizes;
    }
    
    public static int[] getFeatureSize(final String appDirPath, final String eventFileName, final String userFeatureMapFileName, final String itemFeatureMapFileName) {
        String filePath = "";
        BufferedReader br = null;
        String line = "";
        String[] container = null;
        String lastLine = "";
        int Pu = 0;
        int Pv = 0;
        int Pe = 0;
        filePath = String.valueOf(appDirPath) + File.separator + userFeatureMapFileName;
        if (!new File(filePath).exists()) {
            System.err.println(String.format("File %s doesn't exist.\n", filePath));
            exit(1);
        }
        try {
            br = new BufferedReader(new FileReader(filePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        try {
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                lastLine = line;
            }
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        container = lastLine.split("\t");
        Pu = Integer.parseInt(container[0]) + 1;
        filePath = String.valueOf(appDirPath) + File.separator + itemFeatureMapFileName;
        if (!new File(filePath).exists()) {
            System.err.println(String.format("File %s doesn't exist.\n", filePath));
            exit(1);
        }
        try {
            br = new BufferedReader(new FileReader(filePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        try {
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                lastLine = line;
            }
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        container = lastLine.split("\t");
        Pv = Integer.parseInt(container[0]) + 1;
        filePath = String.valueOf(appDirPath) + File.separator + eventFileName;
        if (!new File(filePath).exists()) {
            System.err.println(String.format("File %s doesn't exist.\n", filePath));
            exit(1);
        }
        try {
            br = new BufferedReader(new FileReader(filePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        try {
            while ((line = br.readLine()) != null && line.isEmpty()) {}
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        container = line.split("\t");
        Pe = container.length - 3;
        final int[] featureSizes = { Pu, Pv, Pe };
        return featureSizes;
    }
    
    public static void saveString(final String filePath, final String content) {
        PrintWriter pw = null;
        final boolean autoFlush = true;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), autoFlush);
        }
        catch (IOException e) {
            e.printStackTrace();
            exit(1);
        }
        pw.print(content);
        pw.close();
    }
    
    public static int[] getNumSelfEdge(final String GEOIndexFilePath, final String YCTIndexFilePath) {
        BufferedReader br = null;
        String line = "";
        String lastLine = "";
        int numSelfEdgeUser = 0;
        int numSelfEdgeItem = 0;
        try {
            br = new BufferedReader(new FileReader(GEOIndexFilePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        try {
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                lastLine = line;
            }
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        numSelfEdgeUser = Integer.parseInt(lastLine.trim()) + 1;
        try {
            br = new BufferedReader(new FileReader(YCTIndexFilePath));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            exit(1);
        }
        try {
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                lastLine = line;
            }
            br.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        numSelfEdgeItem = Integer.parseInt(lastLine.trim()) + 1;
        return new int[] { numSelfEdgeUser, numSelfEdgeItem };
    }
    
    public static String executeCommand(final String command) {
        final StringBuffer output = new StringBuffer();
        try {
            final Process p = Runtime.getRuntime().exec(command);
            final BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line = "";
            while ((line = reader.readLine()) != null) {
                output.append(String.valueOf(line) + "\n");
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return output.toString();
    }
    
    public static void saveMeasures(final String appDirPath, final String fileName, final double[] measures) {
        final String filePath = String.valueOf(appDirPath) + File.separator + fileName;
        PrintWriter pw = null;
        final boolean autoFlush = true;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), autoFlush);
        }
        catch (IOException e) {
            e.printStackTrace();
            exit(1);
        }
        final double RMSE = measures[0];
        final double MAP = measures[1];
        final double MRR = measures[2];
        final double MP10 = measures[3];
        pw.printf("RMSE: %.8g\n", RMSE);
        pw.printf("MAP: %.8g\n", MAP);
        pw.printf("MRR: %.8g\n", MRR);
        pw.printf("MP@10: %.8g\n", MP10);
        pw.close();
    }
    
    public static void loadMap(final TreeMap<Integer, Integer> map, final String filePath) {
        BufferedReader br = null;
        String line = "";
        String[] container = null;
        try {
            br = new BufferedReader(new FileReader(filePath));
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                container = line.split("\t");
                map.put(Integer.valueOf(container[0]), Integer.valueOf(container[1]));
            }
            br.close();
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
    }
    
    public static double[] predict(final Data testData, final double[] Yij_pred, final TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap, final TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap) {
        final double[] Yij = testData.Yij;
        final HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = testData.User2EventIndexSetMap;
        if (!Utility.drawMAPCurve) {
            System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
        }
        final int T = testData.T;
        final int M = testData.M;
        double RMSE = 0.0;
        for (int e = 0; e < T; ++e) {
            final double y = Yij[e];
            final double y_pred = Yij_pred[e];
            final double error = y - y_pred;
            RMSE += error * error;
        }
        RMSE /= T;
        RMSE = Math.sqrt(RMSE);
        double MAP = 0.0;
        double MRR = 0.0;
        double MP10 = 0.0;
        for (int i = 0; i < M; ++i) {
            final LinkedList<Integer> eventIdxList = User2EventIndexSetMap.get(i);
            final double[] scoresOfItems4UserI = new double[eventIdxList.size()];
            final double[] groundTruth4UserI = new double[eventIdxList.size()];
            int k = 0;
            for (final int eventIdx : eventIdxList) {
                scoresOfItems4UserI[k] = Yij_pred[eventIdx];
                groundTruth4UserI[k++] = Yij[eventIdx];
            }
            final int[] eventListPosition4RankAt = ArrayOperator.sort(scoresOfItems4UserI, "descend");
            double AP_i = 0.0;
            double RR_i = 0.0;
            double P10_i = 0.0;
            int numRelItems = 0;
            for (k = 0; k < eventIdxList.size(); ++k) {
                final int eventListPos = eventListPosition4RankAt[k];
                if (groundTruth4UserI[eventListPos] == 1.0) {
                    ++numRelItems;
                    AP_i += numRelItems / (k + 1.0);
                    if (numRelItems == 1) {
                        RR_i += 1.0 / (k + 1.0);
                    }
                    if (k < 10) {
                        P10_i += 0.1;
                    }
                }
            }
            AP_i /= numRelItems;
            MAP += AP_i;
            MRR += RR_i;
            MP10 += P10_i;
        }
        MAP /= M;
        MRR /= M;
        MP10 /= M;
        if (!Utility.drawMAPCurve) {
            System.out.printf("RMSE: %.6g\n", RMSE);
            System.out.printf("MAP: %.6g\n", MAP);
            System.out.printf("MRR: %.6g\n", MRR);
            System.out.printf("MP@10: %6g\n", MP10);
        }
        final double[] res = { RMSE, MAP, MRR, MP10 };
        return res;
    }
    
    public static double[] predict(final int[] Yij, final double[] Yij_pred, final HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap) {
        if (!Utility.drawMAPCurve) {
            System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
        }
        final int T = Yij.length;
        final int M = TestUser2EventIndexSetMap.size();
        double RMSE = 0.0;
        for (int e = 0; e < T; ++e) {
            final double y = Yij[e];
            final double y_pred = Yij_pred[e];
            final double error = y - y_pred;
            RMSE += error * error;
        }
        RMSE /= T;
        RMSE = Math.sqrt(RMSE);
        double MAP = 0.0;
        double MRR = 0.0;
        double MP10 = 0.0;
        for (int i = 0; i < M; ++i) {
            final LinkedList<Integer> eventIdxList = TestUser2EventIndexSetMap.get(i);
            final double[] scoresOfItems4UserI = new double[eventIdxList.size()];
            final double[] groundTruth4UserI = new double[eventIdxList.size()];
            int k = 0;
            for (final int eventIdx : eventIdxList) {
                scoresOfItems4UserI[k] = Yij_pred[eventIdx];
                groundTruth4UserI[k++] = Yij[eventIdx];
            }
            final int[] eventListPosition4RankAt = ArrayOperator.sort(scoresOfItems4UserI, "descend");
            double AP_i = 0.0;
            double RR_i = 0.0;
            double P10_i = 0.0;
            int numRelItems = 0;
            for (k = 0; k < eventIdxList.size(); ++k) {
                final int eventListPos = eventListPosition4RankAt[k];
                if (groundTruth4UserI[eventListPos] == 1.0) {
                    ++numRelItems;
                    AP_i += numRelItems / (k + 1.0);
                    if (numRelItems == 1) {
                        RR_i += 1.0 / (k + 1.0);
                    }
                    if (k < 10) {
                        P10_i += 0.1;
                    }
                }
            }
            AP_i /= numRelItems;
            MAP += AP_i;
            MRR += RR_i;
            MP10 += P10_i;
        }
        MAP /= M;
        MRR /= M;
        MP10 /= M;
        if (!Utility.drawMAPCurve) {
            System.out.printf("RMSE: %.6g\n", RMSE);
            System.out.printf("MAP: %.6g\n", MAP);
            System.out.printf("MRR: %.6g\n", MRR);
            System.out.printf("MP@10: %6g\n", MP10);
        }
        final double[] res = { RMSE, MAP, MRR, MP10 };
        return res;
    }
    
    public static double[] predictColdStart(final Data testData, final double[] Yij_pred, final TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap, final TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap) {
        final double[] Yij = testData.Yij;
        final int[] UserIndices = testData.UserIndices;
        final HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap = testData.User2EventIndexSetMap;
        final int M = testData.M;
        final int T = testData.T;
        if (!Utility.drawMAPCurve) {
            System.out.println("\nCold start setting:");
            System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
        }
        int numColdStart = 0;
        double RMSE = 0.0;
        for (int e = 0; e < T; ++e) {
            final double y = Yij[e];
            final double y_pred = Yij_pred[e];
            final int i = UserIndices[e];
            final int trainIdx = TestIdx2TrainIdxUserMap.get(i);
            if (trainIdx == -1) {
                ++numColdStart;
                final double error = y - y_pred;
                RMSE += error * error;
            }
        }
        RMSE /= numColdStart;
        RMSE = Math.sqrt(RMSE);
        int numNewUsers = 0;
        double MAP = 0.0;
        double MRR = 0.0;
        double MP10 = 0.0;
        for (int j = 0; j < M; ++j) {
            final int trainIdx2 = TestIdx2TrainIdxUserMap.get(j);
            if (trainIdx2 == -1) {
                ++numNewUsers;
                final LinkedList<Integer> eventIdxList = User2EventIndexSetMap.get(j);
                final double[] scoresOfItems4UserI = new double[eventIdxList.size()];
                final double[] groundTruth4UserI = new double[eventIdxList.size()];
                int k = 0;
                for (final int eventIdx : eventIdxList) {
                    scoresOfItems4UserI[k] = Yij_pred[eventIdx];
                    groundTruth4UserI[k++] = Yij[eventIdx];
                }
                final int[] eventListPosition4RankAt = ArrayOperator.sort(scoresOfItems4UserI, "descend");
                double AP_i = 0.0;
                double RR_i = 0.0;
                double P10_i = 0.0;
                int numRelItems = 0;
                for (k = 0; k < eventIdxList.size(); ++k) {
                    final int eventListPos = eventListPosition4RankAt[k];
                    if (groundTruth4UserI[eventListPos] == 1.0) {
                        ++numRelItems;
                        AP_i += numRelItems / (k + 1.0);
                        if (numRelItems == 1) {
                            RR_i += 1.0 / (k + 1.0);
                        }
                        if (k < 10) {
                            P10_i += 0.1;
                        }
                    }
                }
                AP_i /= numRelItems;
                MAP += AP_i;
                MRR += RR_i;
                MP10 += P10_i;
            }
        }
        MAP /= numNewUsers;
        MRR /= numNewUsers;
        MP10 /= numNewUsers;
        if (!Utility.drawMAPCurve) {
            System.out.printf("RMSE: %.6g\n", RMSE);
            System.out.printf("MAP: %.6g\n", MAP);
            System.out.printf("MRR: %.6g\n", MRR);
            System.out.printf("MP@10: %6g\n", MP10);
        }
        final double[] res = { RMSE, MAP, MRR, MP10 };
        return res;
    }
    
    public static double[] predictColdStart(final int[] Yij, final double[] Yij_pred, final int[] TestUserIndices, final HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap, final TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap, final TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap) {
        final int M = TestUser2EventIndexSetMap.size();
        final int T = Yij.length;
        if (!Utility.drawMAPCurve) {
            System.out.println("\nCold start setting:");
            System.out.println("\nComputing RMSE, MAP, MRR, and MP@10...");
        }
        int numColdStart = 0;
        double RMSE = 0.0;
        for (int e = 0; e < T; ++e) {
            final double y = Yij[e];
            final double y_pred = Yij_pred[e];
            final int i = TestUserIndices[e];
            final int trainIdx = TestIdx2TrainIdxUserMap.get(i);
            if (trainIdx == -1) {
                ++numColdStart;
                final double error = y - y_pred;
                RMSE += error * error;
            }
        }
        RMSE /= numColdStart;
        RMSE = Math.sqrt(RMSE);
        int numNewUsers = 0;
        double MAP = 0.0;
        double MRR = 0.0;
        double MP10 = 0.0;
        for (int j = 0; j < M; ++j) {
            final int trainIdx2 = TestIdx2TrainIdxUserMap.get(j);
            if (trainIdx2 == -1) {
                ++numNewUsers;
                final LinkedList<Integer> eventIdxList = TestUser2EventIndexSetMap.get(j);
                final double[] scoresOfItems4UserI = new double[eventIdxList.size()];
                final double[] groundTruth4UserI = new double[eventIdxList.size()];
                int k = 0;
                for (final int eventIdx : eventIdxList) {
                    scoresOfItems4UserI[k] = Yij_pred[eventIdx];
                    groundTruth4UserI[k++] = Yij[eventIdx];
                }
                final int[] eventListPosition4RankAt = ArrayOperator.sort(scoresOfItems4UserI, "descend");
                double AP_i = 0.0;
                double RR_i = 0.0;
                double P10_i = 0.0;
                int numRelItems = 0;
                for (k = 0; k < eventIdxList.size(); ++k) {
                    final int eventListPos = eventListPosition4RankAt[k];
                    if (groundTruth4UserI[eventListPos] == 1.0) {
                        ++numRelItems;
                        AP_i += numRelItems / (k + 1.0);
                        if (numRelItems == 1) {
                            RR_i += 1.0 / (k + 1.0);
                        }
                        if (k < 10) {
                            P10_i += 0.1;
                        }
                    }
                }
                AP_i /= numRelItems;
                MAP += AP_i;
                MRR += RR_i;
                MP10 += P10_i;
            }
        }
        MAP /= numNewUsers;
        MRR /= numNewUsers;
        MP10 /= numNewUsers;
        if (!Utility.drawMAPCurve) {
            System.out.printf("RMSE: %.6g\n", RMSE);
            System.out.printf("MAP: %.6g\n", MAP);
            System.out.printf("MRR: %.6g\n", MRR);
            System.out.printf("MP@10: %6g\n", MP10);
        }
        final double[] res = { RMSE, MAP, MRR, MP10 };
        return res;
    }
}
