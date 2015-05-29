package ml.classification;

import ml.utils.*;
import la.matrix.*;
import java.io.*;
import java.util.*;

public class AdaBoost extends Classifier
{
    private static final long serialVersionUID = 1100546985050582205L;
    int T;
    Classifier[] weakClassifiers;
    double[] alphas;
    
    public static void main(final String[] args) {
        final double[][] data = { { 3.5, 4.4, 1.3 }, { 5.3, 2.2, 0.5 }, { 0.2, 0.3, 4.1 }, { 5.3, 2.2, -1.5 }, { -1.2, 0.4, 3.2 } };
        final int[] labels = { 1, 1, -1, -1, -1 };
        final Matrix X = new DenseMatrix(data);
        final double epsilon = 1.0E-5;
        final Classifier logReg = new LogisticRegression(epsilon);
        logReg.feedData(X);
        logReg.feedLabels(labels);
        logReg.train();
        Matrix Xt = X;
        double accuracy = Classifier.getAccuracy(labels, logReg.predict(Xt));
        Printer.fprintf("Accuracy for logistic regression: %.2f%%\n", 100.0 * accuracy);
        final int T = 10;
        final Classifier[] weakClassifiers = new Classifier[T];
        for (int t = 0; t < 10; ++t) {
            weakClassifiers[t] = new LogisticRegression(epsilon);
        }
        final Classifier adaBoost = new AdaBoost(weakClassifiers);
        adaBoost.feedData(X);
        adaBoost.feedLabels(labels);
        Time.tic();
        adaBoost.train();
        System.out.format("Elapsed time: %.2f seconds.%n", Time.toc());
        Xt = X.copy();
        Printer.display(adaBoost.predictLabelScoreMatrix(Xt));
        Printer.display(Matlab.full(adaBoost.predictLabelMatrix(Xt)));
        Printer.display(adaBoost.predict(Xt));
        accuracy = Classifier.getAccuracy(labels, adaBoost.predict(Xt));
        Printer.fprintf("Accuracy for AdaBoost with logistic regression: %.2f%%\n", 100.0 * accuracy);
        final String modelFilePath = "AdaBoostModel";
        adaBoost.saveModel(modelFilePath);
        final Classifier adaBoost2 = new AdaBoost();
        adaBoost2.loadModel(modelFilePath);
        Printer.display(adaBoost2.predictLabelScoreMatrix(Xt));
        Printer.display(Matlab.full(adaBoost2.predictLabelMatrix(Xt)));
        Printer.display(adaBoost2.predict(Xt));
        accuracy = Classifier.getAccuracy(labels, adaBoost2.predict(Xt));
        Printer.fprintf("Accuracy: %.2f%%\n", 100.0 * accuracy);
    }
    
    public AdaBoost(final Classifier[] weakClassifiers) {
        this.T = weakClassifiers.length;
        this.weakClassifiers = weakClassifiers;
        this.alphas = new double[this.T];
    }
    
    public AdaBoost() {
    }
    
    @Override
    public void loadModel(final String filePath) {
        System.out.println("Loading model...");
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            this.weakClassifiers = (Classifier[])ois.readObject();
            this.T = this.weakClassifiers.length;
            this.alphas = (double[])ois.readObject();
            this.IDLabelMap = (int[])ois.readObject();
            this.nClass = this.IDLabelMap.length;
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
            oos.writeObject(this.weakClassifiers);
            oos.writeObject(this.alphas);
            oos.writeObject(this.IDLabelMap);
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
    public void train() {
        final int d = this.nFeature;
        final int m = this.nExample;
        Matrix Dt = Matlab.times(1.0 / m, Matlab.ones(m, 1));
        Matrix Xt = Matlab.zeros(d, m);
        final Matrix errs = Matlab.zeros(1, this.T);
        final Matrix ets = Matlab.zeros(1, this.T);
        final Matrix Coef = Matlab.diag(Matlab.diag(new double[] { 1.0, -1.0 }));
        final Random generator = new Random();
        final Matrix Y_true_labels = Matlab.mtimes(this.Y, Coef);
        Matrix Yt_LabelMatrix = Matlab.zeros(m, this.nClass);
        for (int t = 1; t <= this.T; ++t) {
            if (t == 1) {
                Xt = this.X.copy();
                Yt_LabelMatrix = this.Y.copy();
            }
            else if (Matlab.sumAll(Dt) == 0.0) {
                Xt = this.X.copy();
                Yt_LabelMatrix = this.Y.copy();
            }
            else {
                for (int i = 1; i <= m; ++i) {
                    double r_i;
                    double s;
                    int j;
                    for (r_i = generator.nextDouble(), s = 0.0, j = 1; j < m && (s > r_i || r_i >= s + Dt.getEntry(j - 1, 0)); s += Dt.getEntry(j - 1, 0), ++j) {}
                    Xt.setRowMatrix(i - 1, this.X.getRowMatrix(j - 1));
                    Yt_LabelMatrix.setRowMatrix(i - 1, this.Y.getRowMatrix(j - 1));
                }
            }
            this.weakClassifiers[t - 1].feedData(Xt);
            this.weakClassifiers[t - 1].feedLabels(Yt_LabelMatrix);
            this.weakClassifiers[t - 1].train();
            final Matrix Y_pred_labels = Matlab.mtimes(this.weakClassifiers[t - 1].predictLabelMatrix(this.X), Coef);
            final Matrix I_err = Matlab.ne(Y_true_labels, Y_pred_labels);
            double et = 0.0;
            for (int k = 0; k < m; ++k) {
                if (I_err.getEntry(k, 0) == 1.0) {
                    et += Dt.getEntry(k, 0);
                }
            }
            ets.setEntry(0, t - 1, et);
            errs.setEntry(0, t - 1, Matlab.sumAll(I_err) / m);
            final double alpha_t = 0.5 * Math.log((1.0 - et) / et);
            this.alphas[t - 1] = alpha_t;
            Dt = Matlab.times(Dt, Matlab.exp(Matlab.times(-alpha_t, Matlab.times(Y_true_labels, Y_pred_labels))));
            final double zt = Matlab.sumAll(Dt);
            if (zt > 0.0) {
                Dt = Matlab.rdivide(Dt, zt);
            }
        }
    }
    
    @Override
    public Matrix predictLabelMatrix(final Matrix Xt) {
        final int m = Xt.getRowDimension();
        Matrix Y_score = Matlab.zeros(m, 1);
        final Matrix Coef = Matlab.diag(Matlab.diag(new double[] { 1.0, -1.0 }));
        for (int t = 1; t <= this.T; ++t) {
            Y_score = Matlab.plus(Y_score, Matlab.times(this.alphas[t - 1], Matlab.mtimes(this.weakClassifiers[t - 1].predictLabelMatrix(Xt), Coef)));
        }
        final Matrix H_final_pred = Matlab.sign(Y_score);
        final Matrix Temp = Matlab.minus(0.5, Matlab.times(0.5, H_final_pred));
        final int[] labelIndices = new int[m];
        for (int i = 0; i < m; ++i) {
            labelIndices[i] = (int)Temp.getEntry(i, 0);
        }
        return Classifier.labelIndexArray2LabelMatrix(labelIndices, this.nClass);
    }
    
    @Override
    public Matrix predictLabelScoreMatrix(final Matrix Xt) {
        final int m = Xt.getRowDimension();
        final Matrix lableScoreMatrix = Matlab.zeros(m, 2);
        Matrix Y_score = Matlab.zeros(m, 1);
        final Matrix Coef = Matlab.diag(Matlab.diag(new double[] { 1.0, -1.0 }));
        for (int t = 1; t <= this.T; ++t) {
            Y_score = Matlab.plus(Y_score, Matlab.times(this.alphas[t - 1], Matlab.mtimes(this.weakClassifiers[t - 1].predictLabelMatrix(Xt), Coef)));
        }
        lableScoreMatrix.setColumnMatrix(0, Y_score);
        lableScoreMatrix.setColumnMatrix(1, Matlab.times(-1.0, Y_score));
        return lableScoreMatrix;
    }
}
