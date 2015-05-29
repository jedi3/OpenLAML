package la.io;

import la.vector.*;
import la.vector.Vector;

import java.util.*;
import java.io.*;

public class DataVectors
{
    public static int IdxStart;
    public Vector[] Vs;
    public int[] Y;
    
    static {
        DataVectors.IdxStart = 1;
    }
    
    public DataVectors() {
        this.Vs = null;
        this.Y = null;
    }
    
    public DataVectors(final Vector[] Vs, final int[] Y) {
        this.Vs = Vs;
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
    
    public static DataVectors readDataVectorsFromStringArray(final ArrayList<String> feaArray) throws InvalidInputDataException {
        DataVectors dataVectors = new DataVectors();
        final List<Integer> vy = new ArrayList<Integer>();
        int max_index = 0;
        int lineNr = 0;
        StringTokenizer labelTokenizer = null;
        StringTokenizer featureTokenizer = null;
        final List<Vector> dataVectorList = new ArrayList<Vector>();
        int exampleIndex = 0;
        int featureIndex = -1;
        double value = 0.0;
        String line = null;
        final Iterator<String> lineIter = feaArray.iterator();
        while (lineIter.hasNext()) {
            line = lineIter.next();
            if (line == null || line.isEmpty()) {
                dataVectorList.add(new SparseVector(max_index + 1));
                ++exampleIndex;
            }
            else {
                final ArrayList<Integer> indexList = new ArrayList<Integer>();
                final ArrayList<Double> valueList = new ArrayList<Double>();
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
                    vy.add(0);
                }
                else {
                    token = featureTokenizer.nextToken();
                    try {
                        vy.add(atoi(token));
                    }
                    catch (NumberFormatException e) {
                        try {
                            vy.add((int)atof(token));
                        }
                        catch (NumberFormatException e4) {
                            throw new InvalidInputDataException("invalid label: " + token, lineNr, e);
                        }
                    }
                }
                final int m = featureTokenizer.countTokens() / 2;
                for (int j = 0; j < m; ++j) {
                    token = featureTokenizer.nextToken();
                    try {
                        featureIndex = atoi(token) - DataVectors.IdxStart;
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
                        indexList.add(featureIndex);
                        valueList.add(value);
                    }
                }
                final int nnz = m;
                final int[] ir = new int[nnz];
                final double[] pr = new double[nnz];
                final Iterator<Integer> indexIter = indexList.iterator();
                final Iterator<Double> valueIter = valueList.iterator();
                for (int k = 0; k < nnz; ++k) {
                    ir[k] = indexIter.next();
                    pr[k] = valueIter.next();
                }
                dataVectorList.add(new SparseVector(ir, pr, nnz, max_index + 1));
                ++exampleIndex;
            }
        }
        final int numRows = exampleIndex;
        final int numColumns = max_index + 1;
        final int[] Y = new int[numRows];
        final Iterator<Integer> iter = vy.iterator();
        int rIdx = 0;
        while (iter.hasNext()) {
            Y[rIdx] = iter.next();
            ++rIdx;
        }
        final Vector[] Vs = new Vector[numRows];
        final Iterator<Vector> dataVectorIter = dataVectorList.iterator();
        for (int i = 0; i < numRows; ++i) {
            Vs[i] = dataVectorIter.next();
            ((SparseVector)Vs[i]).setDim(numColumns);
        }
        dataVectors = new DataVectors(Vs, Y);
        return dataVectors;
    }
    
    public static DataVectors readDataSetFromFile(final String filePath) throws IOException, InvalidInputDataException {
        DataVectors dataVectors = new DataVectors();
        final BufferedReader fp = new BufferedReader(new FileReader(filePath));
        final List<Integer> vy = new ArrayList<Integer>();
        int max_index = 0;
        int lineNr = 0;
        StringTokenizer labelTokenizer = null;
        StringTokenizer featureTokenizer = null;
        final List<Vector> dataVectorList = new ArrayList<Vector>();
        int exampleIndex = 0;
        int featureIndex = -1;
        double value = 0.0;
        String line = null;
        while ((line = fp.readLine()) != null) {
            if (line.isEmpty()) {
                dataVectorList.add(new SparseVector(max_index + 1));
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
                final ArrayList<Integer> indexList = new ArrayList<Integer>();
                final ArrayList<Double> valueList = new ArrayList<Double>();
                if (token.contains(":")) {
                    vy.add(0);
                }
                else {
                    token = featureTokenizer.nextToken();
                    try {
                        vy.add(atoi(token));
                    }
                    catch (NumberFormatException e) {
                        try {
                            vy.add((int)atof(token));
                        }
                        catch (NumberFormatException e4) {
                            fp.close();
                            throw new InvalidInputDataException("invalid label: " + token, lineNr, e);
                        }
                    }
                }
                final int m = featureTokenizer.countTokens() / 2;
                for (int j = 0; j < m; ++j) {
                    token = featureTokenizer.nextToken();
                    try {
                        featureIndex = atoi(token) - DataVectors.IdxStart;
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
                        indexList.add(featureIndex);
                        valueList.add(value);
                    }
                }
                final int nnz = m;
                final int[] ir = new int[nnz];
                final double[] pr = new double[nnz];
                final Iterator<Integer> indexIter = indexList.iterator();
                final Iterator<Double> valueIter = valueList.iterator();
                for (int k = 0; k < nnz; ++k) {
                    ir[k] = indexIter.next();
                    pr[k] = valueIter.next();
                }
                dataVectorList.add(new SparseVector(ir, pr, nnz, max_index + 1));
                ++exampleIndex;
            }
        }
        fp.close();
        final int numRows = exampleIndex;
        final int numColumns = max_index + 1;
        final int[] Y = new int[numRows];
        final Iterator<Integer> iter = vy.iterator();
        int rIdx = 0;
        while (iter.hasNext()) {
            Y[rIdx] = iter.next();
            ++rIdx;
        }
        final Vector[] Vs = new Vector[numRows];
        final Iterator<Vector> dataVectorIter = dataVectorList.iterator();
        for (int i = 0; i < numRows; ++i) {
            Vs[i] = dataVectorIter.next();
            ((SparseVector)Vs[i]).setDim(numColumns);
        }
        dataVectors = new DataVectors(Vs, Y);
        return dataVectors;
    }
}
