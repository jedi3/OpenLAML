package la.io;

import la.matrix.*;
import ml.utils.*;

import java.util.*;
import java.util.regex.*;

import la.vector.*;
import la.vector.Vector;

import java.io.*;

public class IO
{
    public static void main(final String[] args) {
        Printer.fprintf("%d", (int)Math.floor(-0.2));
        final String line = " 1:2.3 ";
        System.out.println(System.lineSeparator());
        final String separator = System.lineSeparator();
        final String separator2 = Printer.sprintf("%n", new Object[0]);
        if (separator.equals(separator2)) {
            System.out.println("System.lineSeparator() == sprintf(\"%n\")");
        }
        final StringTokenizer tokenizer = new StringTokenizer(line, " \t\n\r\f");
        try {
            System.out.println(tokenizer.nextToken());
        }
        catch (NoSuchElementException e3) {
            System.out.println("The line is empty.");
        }
        if (tokenizer.hasMoreTokens()) {
            System.out.println("The line has more tokens.");
            System.out.println(tokenizer.nextToken());
        }
        else {
            System.out.println("The line is empty.");
        }
        final int[] rIndices = { 0, 1, 3, 1, 2, 2, 3, 2, 3 };
        final int[] cIndices = { 0, 0, 0, 1, 1, 2, 2, 3, 3 };
        final double[] values = { 10.0, 3.0, 3.0, 9.0, 7.0, 8.0, 8.0, 7.0, 7.0 };
        final int numRows = 5;
        final int numColumns = 5;
        final int nzmax = rIndices.length;
        final Matrix S = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
        Printer.fprintf("S:\n", new Object[0]);
        Printer.printMatrix(S, 4);
        final Matrix A = Matlab.full(S);
        Printer.fprintf("A:\n", new Object[0]);
        Printer.printMatrix(A, 4);
        final Matrix S2 = Matlab.sparse(A);
        Printer.fprintf("S2:\n", new Object[0]);
        Printer.printMatrix(S2, 4);
        String filePath = null;
        filePath = "SparseMatrix.txt";
        saveMatrix(S, filePath);
        Printer.fprintf("Loaded S:\n", new Object[0]);
        Printer.printMatrix(loadMatrix(filePath));
        DataSet.writeDataSet(S, null, "Dataset.txt");
        DataSet dataSet = null;
        try {
            dataSet = DataSet.readDataSetFromFile("Dataset.txt");
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        catch (InvalidInputDataException e2) {
            e2.printStackTrace();
        }
        Printer.printMatrix(dataSet.X);
        filePath = "DenseMatrix.txt";
        saveMatrix(A, filePath);
        Printer.fprintf("Loaded A:\n", new Object[0]);
        Printer.printMatrix(loadMatrix(filePath));
        final String dataMatrixFilePath = "CNN - DocTermCount.txt";
        final Matrix X = loadMatrixFromDocTermCountFile(dataMatrixFilePath);
        filePath = "CNN-DocTermCountMatrix.txt";
        saveMatrix(X.transpose(), filePath);
        final int dim = 4;
        final Vector V = new SparseVector(dim);
        for (int i = 0; i < dim; ++i) {
            Printer.fprintf("V(%d):\t%.2f\n", i + 1, V.get(i));
        }
        V.set(3, 4.5);
        Printer.fprintf("V(%d):\t%.2f%n", 4, V.get(3));
        V.set(1, 2.3);
        Printer.fprintf("V(%d):\t%.2f%n", 2, V.get(1));
        V.set(1, 3.2);
        Printer.fprintf("V(%d):\t%.2f%n", 2, V.get(1));
        V.set(3, 2.5);
        Printer.fprintf("V(%d):\t%.2f%n", 4, V.get(3));
        Printer.fprintf("V:%n", new Object[0]);
        Printer.disp(V);
        filePath = "SparseVector.txt";
        saveVector(filePath, V);
        Printer.fprintf("Loaded V:\n", new Object[0]);
        Printer.disp(loadVector(filePath));
        filePath = "DenseVector.txt";
        saveVector(filePath, Matlab.full(V));
        Printer.fprintf("Loaded V:\n", new Object[0]);
        Printer.disp(loadVector(filePath));
    }
    
    public static void save(final String filePath, final Matrix A) {
        saveMatrix(filePath, A);
    }
    
    public static void save(final Matrix A, final String filePath) {
        saveMatrix(A, filePath);
    }
    
    public static void saveMatrix(final Matrix A, final String filePath) {
        if (A instanceof DenseMatrix) {
            saveDenseMatrix(A, filePath);
        }
        else {
            saveSparseMatrix(A, filePath);
        }
    }
    
    public static void saveMatrix(final String filePath, final Matrix A) {
        if (A instanceof DenseMatrix) {
            saveDenseMatrix(A, filePath);
        }
        else {
            saveSparseMatrix(A, filePath);
        }
    }
    
    public static void saveDenseMatrix(final Matrix A, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new FileWriter(filePath));
        }
        catch (IOException e) {
            System.out.println("IO error for creating file: " + filePath);
        }
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final double[][] data = ((DenseMatrix)A).getData();
        double[] rowData = null;
        final StringBuilder strBuilder = new StringBuilder(200);
        for (int i = 0; i < nRow; ++i) {
            strBuilder.setLength(0);
            rowData = data[i];
            for (final double v : rowData) {
                final int rv = (int)Math.round(v);
                if (v != rv) {
                    strBuilder.append(String.format("%.8g\t", v));
                }
                else {
                    strBuilder.append(String.format("%d\t", rv));
                }
            }
            pw.println(strBuilder.toString().trim());
        }
        if (!pw.checkError()) {
            pw.close();
            System.out.println("Data matrix file written: " + filePath + System.getProperty("line.separator"));
        }
        else {
            pw.close();
            System.err.println("Print stream has encountered an error!");
        }
    }
    
    public static void saveDenseMatrix(final String filePath, final Matrix A) {
        saveDenseMatrix(A, filePath);
    }
    
    public static void saveSparseMatrix(final Matrix A, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new FileWriter(filePath));
        }
        catch (IOException e) {
            System.out.println("IO error for creating file: " + filePath);
        }
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        pw.printf("numRows: %d%n", nRow);
        pw.printf("numColumns: %d%n", nCol);
        if (A instanceof SparseMatrix) {
            final int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            final double[] pr = ((SparseMatrix)A).getPr();
            int rIdx = -1;
            int cIdx = -1;
            double value = 0.0;
            for (int j = 0; j < nCol; ++j) {
                cIdx = j + 1;
                for (int k = jc[j]; k < jc[j + 1]; ++k) {
                    rIdx = ir[k] + 1;
                    value = pr[k];
                    if (value != 0.0) {
                        final int rv = (int)Math.round(value);
                        if (value != rv) {
                            pw.printf("%d %d %.8g%n", rIdx, cIdx, value);
                        }
                        else {
                            pw.printf("%d %d %d%n", rIdx, cIdx, rv);
                        }
                    }
                }
            }
        }
        if (!pw.checkError()) {
            pw.close();
            System.out.println("Data matrix file written: " + filePath + System.getProperty("line.separator"));
        }
        else {
            pw.close();
            System.err.println("Print stream has encountered an error!");
        }
    }
    
    public static void saveSparseMatrix(final String filePath, final Matrix A) {
        saveSparseMatrix(A, filePath);
    }
    
    public static Matrix loadMatrix(final String filePath) {
        Matrix M = null;
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(filePath));
        }
        catch (FileNotFoundException e) {
            System.out.println("Cannot open file: " + filePath);
            e.printStackTrace();
        }
        String line = "";
        int ind = 0;
        boolean isSparseMatrix = false;
        try {
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    if (Pattern.matches("numRows:[\\s]*([\\d]+)", line)) {
                        isSparseMatrix = true;
                        break;
                    }
                    if (Pattern.matches("numColumns:[\\s]*([\\d]+)", line)) {
                        isSparseMatrix = true;
                        break;
                    }
                    if (Pattern.matches("[(]?([\\d]+)[,] ([\\d]+)[)]?[:]? ([-\\d.]+)", line)) {
                        isSparseMatrix = true;
                        break;
                    }
                    if (++ind == 2) {
                        break;
                    }
                    continue;
                }
            }
            br.close();
            if (isSparseMatrix) {
                M = loadSparseMatrix(filePath);
            }
            else {
                M = loadDenseMatrix(filePath);
            }
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        return M;
    }
    
    public static DenseMatrix loadDenseMatrix(final String filePath) {
        BufferedReader textIn = null;
        try {
            textIn = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String line = null;
        final ArrayList<double[]> denseArr = new ArrayList<double[]>();
        try {
            while ((line = textIn.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    final String[] strArr = line.split("[\t ]");
                    final double[] vec = new double[strArr.length];
                    for (int i = 0; i < strArr.length; ++i) {
                        vec[i] = Double.parseDouble(strArr[i]);
                    }
                    denseArr.add(vec);
                }
            }
            textIn.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        final int nRow = denseArr.size();
        final double[][] data = new double[nRow][];
        final Iterator<double[]> iter = denseArr.iterator();
        int rIdx = 0;
        while (iter.hasNext()) {
            data[rIdx++] = iter.next();
        }
        return new DenseMatrix(data);
    }
    
    public static SparseMatrix loadSparseMatrix(final String filePath) {
        Pattern pattern = null;
        BufferedReader br = null;
        Matcher matcher = null;
        int rIdx = 0;
        int cIdx = 0;
        int nzmax = 0;
        double value = 0.0;
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        try {
            br = new BufferedReader(new FileReader(filePath));
        }
        catch (FileNotFoundException e) {
            System.err.println("Cannot open file: " + filePath);
            e.printStackTrace();
            System.exit(1);
        }
        int numRows = -1;
        int numColumns = -1;
        int estimatedNumRows = -1;
        int estimatedNumCols = -1;
        int ind = 0;
        try {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    if (Pattern.matches("numRows:[\\s]*([\\d]+)", line)) {
                        matcher = Pattern.compile("numRows:[\\s]*([\\d]+)").matcher(line);
                        if (matcher.find()) {
                            numRows = Integer.parseInt(matcher.group(1));
                        }
                    }
                    else if (Pattern.matches("numColumns:[\\s]*([\\d]+)", line)) {
                        matcher = Pattern.compile("numColumns:[\\s]*([\\d]+)").matcher(line);
                        if (matcher.find()) {
                            numColumns = Integer.parseInt(matcher.group(1));
                        }
                    }
                    if (++ind == 2) {
                        break;
                    }
                    continue;
                }
            }
        }
        catch (IOException e2) {
            e2.printStackTrace();
            System.exit(1);
        }
        pattern = Pattern.compile("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)");
        try {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    matcher = pattern.matcher(line);
                    if (!matcher.find()) {
                        continue;
                    }
                    rIdx = Integer.parseInt(matcher.group(1)) - 1;
                    cIdx = Integer.parseInt(matcher.group(2)) - 1;
                    value = Double.parseDouble(matcher.group(3));
                    if (value != 0.0) {
                        map.put(Pair.of(cIdx, rIdx), value);
                        ++nzmax;
                    }
                    if (estimatedNumRows < rIdx + 1) {
                        estimatedNumRows = rIdx + 1;
                    }
                    if (estimatedNumCols >= cIdx + 1) {
                        continue;
                    }
                    estimatedNumCols = cIdx + 1;
                }
            }
            br.close();
        }
        catch (NumberFormatException e3) {
            e3.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
            System.exit(1);
        }
        numRows = ((numRows == -1) ? estimatedNumRows : numRows);
        numColumns = ((numColumns == -1) ? estimatedNumCols : numColumns);
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int k = 0;
        jc[0] = 0;
        int currentColumn = 0;
        for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
            rIdx = entry.getKey().second;
            cIdx = entry.getKey().first;
            pr[k] = entry.getValue();
            ir[k] = rIdx;
            while (currentColumn < cIdx) {
                jc[currentColumn + 1] = k;
                ++currentColumn;
            }
            ++k;
        }
        while (currentColumn < numColumns) {
            jc[currentColumn + 1] = k;
            ++currentColumn;
        }
        jc[numColumns] = k;
        return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
    
    public static SparseMatrix loadMatrixFromDocTermCountFile(final String docTermCountFilePath) {
        Pattern pattern = null;
        BufferedReader br = null;
        Matcher matcher = null;
        int docID = 0;
        int featureID = 0;
        double value = 0.0;
        int nDoc = 0;
        int nFeature = 0;
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        pattern = Pattern.compile("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)");
        try {
            br = new BufferedReader(new FileReader(docTermCountFilePath));
        }
        catch (FileNotFoundException e) {
            System.out.println("Cannot open file: " + docTermCountFilePath);
            e.printStackTrace();
            return null;
        }
        int nzmax = 0;
        SparseMatrix res = null;
        try {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    matcher = pattern.matcher(line);
                    if (!matcher.find()) {
                        System.out.println("Data format for the docTermCountFile should be: (DocID, featureID): value");
                        System.exit(0);
                    }
                    docID = Integer.parseInt(matcher.group(1));
                    featureID = Integer.parseInt(matcher.group(2));
                    value = Double.parseDouble(matcher.group(3));
                    if (nFeature < featureID) {
                        nFeature = featureID;
                    }
                    if (nDoc < docID) {
                        nDoc = docID;
                    }
                    if (value == 0.0) {
                        continue;
                    }
                    map.put(Pair.of(docID - 1, featureID - 1), value);
                    ++nzmax;
                }
            }
            br.close();
            final int numRows = nFeature;
            final int numColumns = nDoc;
            final int[] ir = new int[nzmax];
            final int[] jc = new int[numColumns + 1];
            final double[] pr = new double[nzmax];
            int rIdx = -1;
            int cIdx = -1;
            int k = 0;
            jc[0] = 0;
            int currentColumn = 0;
            for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
                rIdx = entry.getKey().second;
                cIdx = entry.getKey().first;
                pr[k] = entry.getValue();
                ir[k] = rIdx;
                while (currentColumn < cIdx) {
                    jc[currentColumn + 1] = k;
                    ++currentColumn;
                }
                ++k;
            }
            while (currentColumn < numColumns) {
                jc[currentColumn + 1] = k;
                ++currentColumn;
            }
            jc[numColumns] = k;
            res = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        }
        catch (NumberFormatException e2) {
            e2.printStackTrace();
        }
        catch (IOException e3) {
            e3.printStackTrace();
        }
        return res;
    }
    
    public static Matrix docTermCountArray2Matrix(final ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {
        int featureID = 0;
        double value = 0.0;
        int nDoc = 0;
        int nFeature = 0;
        int docID = 0;
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        int nzmax = 0;
        Matrix res = null;
        Iterator<TreeMap<Integer, Integer>> iter = docTermCountArray.iterator();
        TreeMap<Integer, Integer> feature = null;
        iter = docTermCountArray.iterator();
        while (iter.hasNext()) {
            feature = iter.next();
            ++docID;
            final Iterator<Integer> iterator = feature.keySet().iterator();
            while (iterator.hasNext()) {
                final int termID = featureID = iterator.next();
                value = feature.get(termID);
                if (nFeature < featureID) {
                    nFeature = featureID;
                }
                if (nDoc < docID) {
                    nDoc = docID;
                }
                if (value != 0.0) {
                    map.put(Pair.of(docID - 1, featureID - 1), value);
                    ++nzmax;
                }
            }
        }
        final int numRows = nFeature;
        final int numColumns = nDoc;
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        int k = 0;
        jc[0] = 0;
        int currentColumn = 0;
        for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
            rIdx = entry.getKey().second;
            cIdx = entry.getKey().first;
            pr[k] = entry.getValue();
            ir[k] = rIdx;
            while (currentColumn < cIdx) {
                jc[currentColumn + 1] = k;
                ++currentColumn;
            }
            ++k;
        }
        while (currentColumn < numColumns) {
            jc[currentColumn + 1] = k;
            ++currentColumn;
        }
        jc[numColumns] = k;
        res = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        return res;
    }
    
    public static void save(final String filePath, final Vector V) {
        saveVector(filePath, V);
    }
    
    public static void save(final Vector V, final String filePath) {
        saveVector(V, filePath);
    }
    
    public static void saveVector(final Vector V, final String filePath) {
        if (V instanceof DenseVector) {
            saveDenseVector((DenseVector)V, filePath);
        }
        else if (V instanceof SparseVector) {
            saveSparseVector((SparseVector)V, filePath);
        }
    }
    
    public static void saveVector(final String filePath, final Vector V) {
        saveVector(V, filePath);
    }
    
    public static void saveDenseVector(final DenseVector V, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            System.out.println("IO error for creating file: " + filePath);
            return;
        }
        final int dim = V.getDim();
        for (final double v : V.getPr()) {
            final int rv = (int)Math.round(v);
            if (v != rv) {
                pw.printf("%.8g%n", v);
            }
            else {
                pw.printf("%d%n", rv);
            }
        }
        if (!pw.checkError()) {
            pw.close();
            System.out.println("Data vector file written: " + filePath + System.getProperty("line.separator"));
        }
        else {
            pw.close();
            System.err.println("Print stream has encountered an error!");
        }
    }
    
    public static void saveSparseVector(final SparseVector V, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            System.out.println("IO error for creating file: " + filePath);
            return;
        }
        final int dim = V.getDim();
        pw.printf("dim: %d%n", dim);
        final int[] ir = V.getIr();
        final double[] pr = V.getPr();
        final int nnz = V.getNNZ();
        int idx = -1;
        double val = 0.0;
        for (int k = 0; k < nnz; ++k) {
            idx = ir[k] + 1;
            val = pr[k];
            if (val != 0.0) {
                final int rv = (int)Math.round(val);
                if (val != rv) {
                    pw.printf("%d %.8g%n", idx, val);
                }
                else {
                    pw.printf("%d %d%n", idx, rv);
                }
            }
        }
        if (!pw.checkError()) {
            pw.close();
            System.out.println("Data vector file written: " + filePath + System.getProperty("line.separator"));
        }
        else {
            pw.close();
            System.err.println("Print stream has encountered an error!");
        }
    }
    
    public static void saveDenseVector(final String filePath, final DenseVector V) {
        saveDenseVector(V, filePath);
    }
    
    public static void saveSparseVector(final String filePath, final SparseVector V) {
        saveSparseVector(V, filePath);
    }
    
    public static Vector loadVector(final String filePath) {
        Vector V = null;
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(filePath));
        }
        catch (FileNotFoundException e) {
            System.out.println("Cannot open file: " + filePath);
            e.printStackTrace();
            return null;
        }
        String line = "";
        int ind = 0;
        boolean isSparseVector = false;
        try {
            while ((line = br.readLine()) != null) {
                if (!line.startsWith("#")) {
                    if (line.trim().isEmpty()) {
                        continue;
                    }
                    if (Pattern.matches("dim:[\\s]*([\\d]+)", line)) {
                        isSparseVector = true;
                        break;
                    }
                    if (++ind == 1) {
                        break;
                    }
                    continue;
                }
            }
            br.close();
            if (isSparseVector) {
                V = loadSparseVector(filePath);
            }
            else {
                V = loadDenseVector(filePath);
            }
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        return V;
    }
    
    public static DenseVector loadDenseVector(final String filePath) {
        BufferedReader textIn = null;
        try {
            textIn = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        }
        catch (FileNotFoundException e) {
            System.out.println("Cannot open file: " + filePath);
            e.printStackTrace();
            return null;
        }
        String line = null;
        final ArrayList<Double> denseArr = new ArrayList<Double>();
        try {
            while ((line = textIn.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    denseArr.add(Double.parseDouble(line));
                }
            }
            textIn.close();
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        final int dim = denseArr.size();
        final double[] pr = new double[dim];
        final Iterator<Double> iter = denseArr.iterator();
        int idx = 0;
        while (iter.hasNext()) {
            pr[idx++] = iter.next();
        }
        return new DenseVector(pr);
    }
    
    public static SparseVector loadSparseVector(final String filePath) {
        Pattern pattern = null;
        BufferedReader br = null;
        Matcher matcher = null;
        int idx = 0;
        int nnz = 0;
        double val = 0.0;
        final TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();
        try {
            br = new BufferedReader(new FileReader(filePath));
        }
        catch (FileNotFoundException e) {
            System.err.println("Cannot open file: " + filePath);
            e.printStackTrace();
            return null;
        }
        int dim = -1;
        int estimatedDim = -1;
        int ind = 0;
        try {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    if (Pattern.matches("dim:[\\s]*([\\d]+)", line)) {
                        matcher = Pattern.compile("dim:[\\s]*([\\d]+)").matcher(line);
                        if (matcher.find()) {
                            dim = Integer.parseInt(matcher.group(1));
                        }
                    }
                    if (++ind == 1) {
                        break;
                    }
                    continue;
                }
            }
        }
        catch (IOException e2) {
            e2.printStackTrace();
            System.exit(1);
        }
        pattern = Pattern.compile("[(]?([\\d]+)[)]?[:]? ([-\\d.]+)");
        try {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("#")) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    matcher = pattern.matcher(line);
                    if (!matcher.find()) {
                        continue;
                    }
                    idx = Integer.parseInt(matcher.group(1)) - 1;
                    val = Double.parseDouble(matcher.group(2));
                    if (val != 0.0) {
                        map.put(idx, val);
                        ++nnz;
                    }
                    if (estimatedDim >= idx + 1) {
                        continue;
                    }
                    estimatedDim = idx + 1;
                }
            }
            br.close();
        }
        catch (NumberFormatException e3) {
            e3.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
            System.exit(1);
        }
        dim = ((dim == -1) ? estimatedDim : dim);
        final int[] ir = new int[nnz];
        final double[] pr = new double[nnz];
        int k = 0;
        for (final Map.Entry<Integer, Double> entry : map.entrySet()) {
            idx = entry.getKey();
            ir[k] = idx;
            pr[k] = entry.getValue();
            ++k;
        }
        return new SparseVector(ir, pr, nnz, dim);
    }
}
