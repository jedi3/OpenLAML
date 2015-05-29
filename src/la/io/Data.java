package la.io;

import la.matrix.*;
import ml.utils.*;
import java.util.*;
import java.io.*;

public class Data
{
    public static int IdxStart;
    public Matrix X;
    public double[] Y;
    
    static {
        Data.IdxStart = 1;
    }
    
    public Data() {
        this.X = null;
        this.Y = null;
    }
    
    public Data(final Matrix X, final double[] Y) {
        this.X = X;
        this.Y = Y;
    }
    
    static double atof(final String s) {
        if (s == null || s.length() < 1) {
            throw new IllegalArgumentException("Can't convert empty string to integer");
        }
        final double d = Double.parseDouble(s);
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            throw new IllegalArgumentException("NaN or Infinity in input: " + s);
        }
        return d;
    }
    
    static int atoi(String s) throws NumberFormatException {
        if (s == null || s.length() < 1) {
            throw new IllegalArgumentException("Can't convert empty string to integer");
        }
        if (s.charAt(0) == '+') {
            s = s.substring(1);
        }
        return Integer.parseInt(s);
    }
    
    public static void writeDataSet(final Matrix X, final double[] Y, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new FileWriter(filePath));
        }
        catch (IOException e) {
            System.out.println("IO error for creating file: " + filePath);
        }
        final int numRows = X.getRowDimension();
        final int numColumns = X.getColumnDimension();
        if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            final double[] pr = ((SparseMatrix)X).getPr();
            final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
            for (int i = 0; i < numRows; ++i) {
                if (Y != null) {
                    pw.printf("%s\t", Y[i]);
                }
                for (int k = jr[i]; k < jr[i + 1]; ++k) {
                    final int j = ic[k];
                    final double v = pr[valCSRIndices[k]];
                    pw.printf("%d:%.8g ", j + Data.IdxStart, v);
                }
                pw.println();
            }
        }
        else if (X instanceof DenseMatrix) {
            final double[][] data = X.getData();
            for (int l = 0; l < numRows; ++l) {
                if (Y != null) {
                    pw.printf("%s\t", Y[l]);
                }
                final double[] row = data[l];
                for (int m = 0; m < numColumns; ++m) {
                    final double v2 = row[m];
                    pw.printf("%d:%.8g ", m + Data.IdxStart, v2);
                }
                pw.println();
            }
        }
        if (!pw.checkError()) {
            pw.close();
            System.out.println("Dataset file written: " + filePath + System.getProperty("line.separator"));
        }
        else {
            pw.close();
            System.err.println("Print stream has encountered an error!");
        }
    }
    
    public static ArrayList<String> writeDataSet(final Matrix X, final double[] Y) {
        final ArrayList<String> res = new ArrayList<String>();
        final int numRows = X.getRowDimension();
        final int numColumns = X.getColumnDimension();
        final StringBuilder sb = new StringBuilder();
        if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            final double[] pr = ((SparseMatrix)X).getPr();
            final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
            for (int i = 0; i < numRows; ++i) {
                sb.setLength(0);
                if (Y != null) {
                    sb.append(Printer.sprintf("%s\t", Y[i]));
                }
                for (int k = jr[i]; k < jr[i + 1]; ++k) {
                    final int j = ic[k];
                    final double v = pr[valCSRIndices[k]];
                    sb.append(Printer.sprintf("%d:%.8g ", j + Data.IdxStart, v));
                }
                res.add(sb.toString());
            }
        }
        else if (X instanceof DenseMatrix) {
            final double[][] data = X.getData();
            for (int l = 0; l < numRows; ++l) {
                sb.setLength(0);
                if (Y != null) {
                    sb.append(Printer.sprintf("%s\t", Y[l]));
                }
                final double[] row = data[l];
                for (int m = 0; m < numColumns; ++m) {
                    final double v2 = row[m];
                    sb.append(Printer.sprintf("%d:%.8g ", m + Data.IdxStart, v2));
                }
                res.add(sb.toString());
            }
        }
        return res;
    }
    
    public static Data readDataSetFromStringArray(final ArrayList<String> feaArray) throws InvalidInputDataException {
        Data dataSet = new Data();
        final List<Double> vy = new ArrayList<Double>();
        int max_index = 0;
        int lineNr = 0;
        StringTokenizer labelTokenizer = null;
        StringTokenizer featureTokenizer = null;
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        int exampleIndex = 0;
        int featureIndex = -1;
        double value = 0.0;
        int nzmax = 0;
        String line = null;
        final Iterator<String> lineIter = feaArray.iterator();
        while (lineIter.hasNext()) {
            line = lineIter.next();
            if (line == null) {
                continue;
            }
            ++lineNr;
            labelTokenizer = new StringTokenizer(line, " \t\n\r\f");
            featureTokenizer = new StringTokenizer(line, " \t\n\r\f:");
            String token;
            try {
                token = labelTokenizer.nextToken();
            }
            catch (NoSuchElementException e3) {
                vy.add(0.0);
                ++exampleIndex;
                continue;
            }
            if (token.contains(":")) {
                vy.add(0.0);
            }
            else {
                token = featureTokenizer.nextToken();
                try {
                    vy.add(atof(token));
                }
                catch (NumberFormatException e) {
                    try {
                        vy.add(atof(token));
                    }
                    catch (NumberFormatException e4) {
                        throw new InvalidInputDataException("invalid label: " + token, lineNr, e);
                    }
                }
            }
            for (int m = featureTokenizer.countTokens() / 2, j = 0; j < m; ++j) {
                token = featureTokenizer.nextToken();
                try {
                    featureIndex = atoi(token) - Data.IdxStart;
                }
                catch (NumberFormatException e2) {
                    throw new InvalidInputDataException("invalid index: " + token, lineNr, e2);
                }
                if (featureIndex < 0) {
                    throw new InvalidInputDataException("invalid index: " + featureIndex, lineNr);
                }
                token = featureTokenizer.nextToken();
                try {
                    value = atof(token);
                }
                catch (NumberFormatException e2) {
                    throw new InvalidInputDataException("invalid value: " + token, lineNr);
                }
                max_index = Math.max(max_index, featureIndex);
                if (value != 0.0) {
                    map.put(Pair.of(featureIndex, exampleIndex), value);
                    ++nzmax;
                }
            }
            ++exampleIndex;
        }
        final int numRows = exampleIndex;
        final int numColumns = max_index + 1;
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
        final Matrix X = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        final double[] Y = new double[numRows];
        final Iterator<Double> iter = vy.iterator();
        rIdx = 0;
        while (iter.hasNext()) {
            Y[rIdx] = iter.next();
            ++rIdx;
        }
        dataSet = new Data(X, Y);
        return dataSet;
    }
    
    public static Data readDataSet(final ArrayList<String> feaArray) throws InvalidInputDataException {
        return readDataSetFromStringArray(feaArray);
    }
    
    public static Data readDataSetFromFile(final String filePath) throws IOException, InvalidInputDataException {
        Data dataSet = new Data();
        final BufferedReader fp = new BufferedReader(new FileReader(filePath));
        final List<Double> vy = new ArrayList<Double>();
        int max_index = 0;
        int lineNr = 0;
        StringTokenizer labelTokenizer = null;
        StringTokenizer featureTokenizer = null;
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        int exampleIndex = 0;
        int featureIndex = -1;
        double value = 0.0;
        int nzmax = 0;
        String line = null;
        while ((line = fp.readLine()) != null) {
            if (line.isEmpty()) {
                vy.add(0.0);
                ++exampleIndex;
            }
            else {
                ++lineNr;
                labelTokenizer = new StringTokenizer(line, " \t\n\r\f");
                featureTokenizer = new StringTokenizer(line, " \t\n\r\f:");
                String token;
                try {
                    token = labelTokenizer.nextToken();
                }
                catch (NoSuchElementException e3) {
                    continue;
                }
                if (token.contains(":")) {
                    vy.add(0.0);
                }
                else {
                    token = featureTokenizer.nextToken();
                    try {
                        vy.add(atof(token));
                    }
                    catch (NumberFormatException e) {
                        try {
                            vy.add(atof(token));
                        }
                        catch (NumberFormatException e4) {
                            fp.close();
                            throw new InvalidInputDataException("invalid label: " + token, lineNr, e);
                        }
                    }
                }
                for (int m = featureTokenizer.countTokens() / 2, j = 0; j < m; ++j) {
                    token = featureTokenizer.nextToken();
                    try {
                        featureIndex = atoi(token) - Data.IdxStart;
                    }
                    catch (NumberFormatException e2) {
                        fp.close();
                        throw new InvalidInputDataException("invalid index: " + token, filePath, lineNr, e2);
                    }
                    if (featureIndex < 0) {
                        fp.close();
                        throw new InvalidInputDataException("invalid index: " + featureIndex, filePath, lineNr);
                    }
                    token = featureTokenizer.nextToken();
                    try {
                        value = atof(token);
                    }
                    catch (NumberFormatException e2) {
                        fp.close();
                        throw new InvalidInputDataException("invalid value: " + token, filePath, lineNr);
                    }
                    max_index = Math.max(max_index, featureIndex);
                    if (value != 0.0) {
                        map.put(Pair.of(featureIndex, exampleIndex), value);
                        ++nzmax;
                    }
                }
                ++exampleIndex;
            }
        }
        fp.close();
        final int numRows = exampleIndex;
        final int numColumns = max_index + 1;
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
        final Matrix X = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        final double[] Y = new double[numRows];
        final Iterator<Double> iter = vy.iterator();
        rIdx = 0;
        while (iter.hasNext()) {
            Y[rIdx] = iter.next();
            ++rIdx;
        }
        dataSet = new Data(X, Y);
        return dataSet;
    }
    
    public static Data readDataSet(final String filePath) throws IOException, InvalidInputDataException {
        return readDataSetFromFile(filePath);
    }
}
