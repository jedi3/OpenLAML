package ml.utils;

import la.io.*;
import la.matrix.*;
import la.vector.*;
import java.io.*;
import java.util.*;

public class IO
{
    public static void save(final String filePath, final double[][] A) {
        la.io.IO.saveMatrix(filePath, new DenseMatrix(A));
    }
    
    public static void save(final double[][] A, final String filePath) {
        la.io.IO.saveMatrix(new DenseMatrix(A), filePath);
    }
    
    public static void save(final String filePath, final double[] V) {
        la.io.IO.saveVector(filePath, new DenseVector(V));
    }
    
    public static void save(final double[] V, final String filePath) {
        la.io.IO.saveVector(new DenseVector(V), filePath);
    }
    
    public static void save(final String filePath, final int[] V) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            System.out.println("IO error for creating file: " + filePath);
            return;
        }
        for (int i = 0; i < V.length; ++i) {
            pw.printf("%d%n", V[i]);
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
    
    public static void save(final int[] V, final String filePath) {
        save(filePath, V);
    }
    
    public static <K, V> void saveMap(final Map<K, V> map, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            e.printStackTrace();
            Utility.exit(1);
        }
        for (final Map.Entry<K, V> entry : map.entrySet()) {
            pw.print(entry.getKey());
            pw.print('\t');
            pw.println(entry.getValue());
        }
        pw.close();
    }
    
    public static <V> void saveList(final List<V> list, final String filePath) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            e.printStackTrace();
            Utility.exit(1);
        }
        for (final V v : list) {
            pw.println(v);
        }
        pw.close();
    }
    
    public static void saveString(final String filePath, final String content) {
        PrintWriter pw = null;
        final boolean autoFlush = true;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), autoFlush);
        }
        catch (IOException e) {
            e.printStackTrace();
            Utility.exit(1);
        }
        pw.print(content);
        pw.close();
    }
    
    public static void save(final String filePath, final String content) {
        saveString(filePath, content);
    }
    
    public static void saveSpec(final DenseVector V, final String[] spec, final String filePath) {
        saveSpec(V, spec, 4, filePath);
    }
    
    public static void saveSpec(final DenseVector V, final String[] spec, final int p, final String filePath) {
        PrintWriter pw = null;
        final boolean autoFlush = true;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), autoFlush);
        }
        catch (IOException e) {
            e.printStackTrace();
            Utility.exit(1);
        }
        if (V instanceof DenseVector) {
            final int dim = V.getDim();
            final double[] pr = V.getPr();
            for (int k = 0; k < dim; ++k) {
                pw.print("  ");
                final double v = pr[k];
                final int rv = (int)Math.round(v);
                String valueString;
                if (v != rv) {
                    valueString = Printer.sprintf(Printer.sprintf("%%.%df", p), v);
                }
                else {
                    valueString = Printer.sprintf("%d", rv);
                }
                pw.println(Printer.sprintf(Printer.sprintf("%%%ds  %%s", 8 + p - 4), valueString, spec[k]));
            }
            pw.println();
        }
        else {
            System.err.println("The input vector should be a DenseVector instance");
            Utility.exit(1);
        }
        pw.close();
    }
}
