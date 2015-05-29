package ml.classification;

import ml.utils.*;
import la.matrix.*;
import ml.optimization.*;
import java.io.*;

public class MaxEnt extends Classifier
{
    private static final long serialVersionUID = -316339495680314422L;
    private Matrix[] F;
    
    public static void main(final String[] args) {
        final long start = System.currentTimeMillis();
        final double[][][] data = { { { 1.0, 0.0, 0.0 }, { 2.0, 1.0, -1.0 }, { 0.0, 1.0, 2.0 }, { -1.0, 2.0, 1.0 } }, { { 0.0, 2.0, 0.0 }, { 1.0, 0.0, -1.0 }, { 0.0, 1.0, 1.0 }, { -1.0, 3.0, 0.5 } }, { { 0.0, 0.0, 0.8 }, { 2.0, 1.0, -1.0 }, { 1.0, 3.0, 0.0 }, { -0.5, -1.0, 2.0 } }, { { 0.5, 0.0, 0.0 }, { 1.0, 1.0, -1.0 }, { 0.0, 0.5, 1.5 }, { -2.0, 1.5, 1.0 } } };
        final int[] labels = { 1, 2, 3, 1 };
        MaxEnt maxEnt = new MaxEnt();
        maxEnt.feedData(data);
        maxEnt.feedLabels(labels);
        maxEnt.train();
        final double elapsedTime = (System.currentTimeMillis() - start) / 1000.0;
        System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);
        Printer.fprintf("MaxEnt parameters:\n", new Object[0]);
        Printer.display(maxEnt.W);
        final String modelFilePath = "MaxEnt-Model.dat";
        maxEnt.saveModel(modelFilePath);
        maxEnt = new MaxEnt();
        maxEnt.loadModel(modelFilePath);
        Printer.fprintf("Predicted probability matrix:\n", new Object[0]);
        Printer.display(maxEnt.predictLabelScoreMatrix(data));
        Printer.fprintf("Predicted label matrix:\n", new Object[0]);
        Printer.display(Matlab.full(maxEnt.predictLabelMatrix(data)));
        Printer.fprintf("Predicted labels:\n", new Object[0]);
        Printer.display(maxEnt.predict(data));
    }
    
    public void feedData(final double[][][] data) {
        this.F = new Matrix[data.length];
        for (int n = 0; n < data.length; ++n) {
            this.F[n] = new DenseMatrix(data[n]);
        }
        this.nExample = data.length;
        this.nFeature = data[0].length;
        this.nClass = data[0][0].length;
    }
    
    public void feedData(final Matrix[] F) {
        this.F = F;
        this.nExample = F.length;
        this.nFeature = F[0].getRowDimension();
        this.nClass = F[0].getColumnDimension();
    }
    
    @Override
    public void train() {
        Matrix Grad = null;
        Matrix A = null;
        Matrix V = null;
        Matrix G = null;
        double fval = 0.0;
        A = new DenseMatrix(this.nExample, this.nClass);
        this.W = Matlab.zeros(this.nFeature, 1);
        for (int n = 0; n < this.nExample; ++n) {
            A.setRowMatrix(n, this.W.transpose().mtimes(this.F[n]));
        }
        V = Matlab.sigmoid(A);
        for (int n = 0; n < this.nExample; ++n) {
            G = Matlab.rdivide(Matlab.minus(Matlab.mtimes(this.F[n], V.getRowMatrix(n).transpose()), this.F[n].getColumnMatrix(this.labelIDs[n])), this.nExample);
            if (n == 0) {
                Grad = G;
            }
            else {
                Grad = Matlab.plus(Grad, G);
            }
        }
        fval = -Matlab.sumAll(Matlab.log(Matlab.logicalIndexing(V, this.Y))) / this.nExample;
        boolean[] flags = null;
        while (true) {
            flags = LBFGS.run(Grad, fval, this.epsilon, this.W);
            if (flags[0]) {
                break;
            }
            for (int n2 = 0; n2 < this.nExample; ++n2) {
                A.setRowMatrix(n2, this.W.transpose().mtimes(this.F[n2]));
            }
            V = Matlab.sigmoid(A);
            fval = -Matlab.sumAll(Matlab.log(Matlab.logicalIndexing(V, this.Y))) / this.nExample;
            if (!flags[1]) {
                continue;
            }
            for (int n2 = 0; n2 < this.nExample; ++n2) {
                G = Matlab.rdivide(Matlab.minus(Matlab.mtimes(this.F[n2], V.getRowMatrix(n2).transpose()), this.F[n2].getColumnMatrix(this.labelIDs[n2])), this.nExample);
                if (n2 == 0) {
                    Grad = G;
                }
                else {
                    Grad = Matlab.plus(Grad, G);
                }
            }
        }
    }
    
    public int[] predict(final double[][][] data) {
        final int Nt = data.length;
        final Matrix[] Ft = new Matrix[Nt];
        for (int n = 0; n < Nt; ++n) {
            Ft[n] = new DenseMatrix(data[n]);
        }
        return this.predict(Ft);
    }
    
    public int[] predict(final Matrix[] Ft) {
        final Matrix Yt = this.predictLabelScoreMatrix(Ft);
        final int[] labelIndices = Classifier.labelScoreMatrix2LabelIndexArray(Yt);
        final int[] labels = new int[labelIndices.length];
        for (int i = 0; i < labelIndices.length; ++i) {
            labels[i] = this.IDLabelMap[labelIndices[i]];
        }
        return labels;
    }
    
    @Override
    public void loadModel(final String filePath) {
        System.out.println("Loading model...");
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            final MaxEntModel MaxEntModel = (MaxEntModel)ois.readObject();
            this.nClass = MaxEntModel.nClass;
            this.W = MaxEntModel.W;
            this.IDLabelMap = MaxEntModel.IDLabelMap;
            this.nFeature = MaxEntModel.nFeature;
            ois.close();
            System.out.println("Model loaded.");
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        catch (ClassNotFoundException e3) {
            e3.printStackTrace();
        }
    }
    
    @Override
    public void saveModel(final String filePath) {
        final File parentFile = new File(filePath).getParentFile();
        if (parentFile != null && !parentFile.exists()) {
            parentFile.mkdirs();
        }
        try {
            final ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
            oos.writeObject(new MaxEntModel(this.nClass, this.W, this.IDLabelMap));
            oos.close();
            System.out.println("Model saved.");
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
    }
    
    @Override
    public Matrix predictLabelScoreMatrix(final Matrix Ft) {
        return null;
    }
    
    public Matrix predictLabelScoreMatrix(final Matrix[] Ft) {
        final int Nt = Ft.length;
        final int K = Ft[0].getColumnDimension();
        Matrix A = null;
        Matrix V = null;
        A = new DenseMatrix(Nt, K);
        for (int i = 0; i < Nt; ++i) {
            A.setRowMatrix(i, this.W.transpose().mtimes(Ft[i]));
        }
        V = Matlab.sigmoid(A);
        return V;
    }
    
    public Matrix predictLabelScoreMatrix(final double[][][] data) {
        final int Nt = data.length;
        final Matrix[] Ft = new Matrix[Nt];
        for (int n = 0; n < Nt; ++n) {
            Ft[n] = new DenseMatrix(data[n]);
        }
        return this.predictLabelScoreMatrix(Ft);
    }
    
    public Matrix predictLabelMatrix(final Matrix[] Ft) {
        final Matrix Yt = this.predictLabelScoreMatrix(Ft);
        final int[] labelIndices = Classifier.labelScoreMatrix2LabelIndexArray(Yt);
        return Classifier.labelIndexArray2LabelMatrix(labelIndices, this.nClass);
    }
    
    public Matrix predictLabelMatrix(final double[][][] data) {
        final int Nt = data.length;
        final Matrix[] Ft = new Matrix[Nt];
        for (int n = 0; n < Nt; ++n) {
            Ft[n] = new DenseMatrix(data[n]);
        }
        return this.predictLabelMatrix(Ft);
    }
}
