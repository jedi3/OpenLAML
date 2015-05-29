package ml.temporal;

import ml.options.*;
import la.vector.*;
import ml.utils.*;
import ml.optimization.*;
import la.matrix.*;
import java.io.*;

public class ReflectedLogisticRegression extends SurvivalScoreModel
{
    private double rho;
    private Options options;
    
    public static void main(final String[] args) {
    }
    
    public ReflectedLogisticRegression(final double lambda) {
        this.options.lambda = lambda;
    }
    
    public ReflectedLogisticRegression(final Options options) {
        this.options = options;
    }
    
    public void initialize(final double rho0) {
        this.rho = rho0;
    }
    
    @Override
    public void initialize(final double... params) {
        if (params.length == 3) {
            this.rho = params[1];
        }
        else {
            this.rho = params[0];
        }
    }
    
    @Override
    public void train() {
        final int ny = this.Y.getColumnDimension();
        final int maxIter = this.options.maxIter;
        if (Double.isInfinite(this.rho)) {
            this.rho = 1.0;
        }
        final double lambda = this.options.lambda;
        final DenseVector weights = new DenseVector(this.n, 0.0);
        final DenseVector GradWT = new DenseVector(this.n, 0.0);
        this.W = Matlab.ones(this.p, ny);
        final Matrix GradW = this.W.copy();
        final Matrix Theta = Matlab.ones(1, 1);
        Theta.setEntry(0, 0, this.rho);
        double gradTheta = 0.0;
        final Matrix ThetaGrad = Matlab.ones(1, 1);
        final Matrix XW = Matlab.zeros(Matlab.size(this.Y));
        double gval = 0.0;
        double hval = 0.0;
        InPlaceOperator.mtimes(XW, this.X, this.W);
        gval = 0.0;
        for (int i = 0; i < this.n; ++i) {
            final double mu = XW.getEntry(i, 0);
            final double t = this.T.getEntry(i, 0);
            final double g = 1.0 / (1.0 + Math.exp(this.rho * (t - mu)));
            final double y = this.Y.getEntry(i, 0);
            final double e = g - y;
            gval += e * e;
            weights.set(i, (g - y) * this.rho * g * (1.0 - g));
        }
        InPlaceOperator.operate(GradWT, weights, this.X);
        for (int j = 0; j < this.p; ++j) {
            GradW.setEntry(j, 0, GradWT.get(j));
        }
        InPlaceOperator.timesAssign(GradW, 2.0 / this.n);
        gval /= this.n;
        hval = lambda * Matlab.norm(this.W, 1);
        final ProximalMapping proxL1 = new ProxL1(lambda);
        final ProximalMapping proxPlus = new ProxPlus();
        AcceleratedProximalGradient.type = 0;
        boolean[] flags = null;
        final double epsilon = 0.001;
        int k = 0;
        final int APGMaxIter = 1000;
        double fval = 0.0;
        double fval_pre = 0.0;
        int cnt = 0;
        do {
            AcceleratedProximalGradient.prox = proxL1;
            while (true) {
                flags = AcceleratedProximalGradient.run(GradW, gval, hval, epsilon, this.W);
                if (flags[0]) {
                    break;
                }
                gval = 0.0;
                InPlaceOperator.mtimes(XW, this.X, this.W);
                for (int l = 0; l < this.n; ++l) {
                    final double mu2 = XW.getEntry(l, 0);
                    final double t2 = this.T.getEntry(l, 0);
                    final double g2 = 1.0 / (1.0 + Math.exp(this.rho * (t2 - mu2)));
                    final double y2 = this.Y.getEntry(l, 0);
                    final double e2 = g2 - y2;
                    gval += e2 * e2;
                    weights.set(l, (g2 - y2) * this.rho * g2 * (1.0 - g2));
                }
                gval /= this.n;
                hval = lambda * Matlab.norm(this.W, 1);
                if (!flags[1]) {
                    continue;
                }
                if (++k > APGMaxIter) {
                    break;
                }
                InPlaceOperator.operate(GradWT, weights, this.X);
                for (int m = 0; m < this.p; ++m) {
                    GradW.setEntry(m, 0, GradWT.get(m));
                }
                InPlaceOperator.timesAssign(GradW, 2.0 / this.n);
            }
            AcceleratedProximalGradient.prox = proxPlus;
            hval = 0.0;
            gradTheta = 0.0;
            for (int l = 0; l < this.n; ++l) {
                final double mu2 = XW.getEntry(l, 0);
                final double t2 = this.T.getEntry(l, 0);
                final double g2 = 1.0 / (1.0 + Math.exp(this.rho * (t2 - mu2)));
                final double y2 = this.Y.getEntry(l, 0);
                gradTheta += (g2 - y2) * (mu2 - t2) * g2 * (1.0 - g2);
            }
            gradTheta *= 2.0 / this.n;
            ThetaGrad.setEntry(0, 0, gradTheta);
            while (true) {
                flags = AcceleratedProximalGradient.run(ThetaGrad, gval, hval, epsilon, Theta);
                if (flags[0]) {
                    break;
                }
                gval = 0.0;
                gradTheta = 0.0;
                this.rho = Theta.getEntry(0, 0);
                for (int l = 0; l < this.n; ++l) {
                    final double mu2 = XW.getEntry(l, 0);
                    final double t2 = this.T.getEntry(l, 0);
                    final double g2 = 1.0 / (1.0 + Math.exp(this.rho * (t2 - mu2)));
                    final double y2 = this.Y.getEntry(l, 0);
                    final double e2 = g2 - y2;
                    gval += e2 * e2;
                    gradTheta += (g2 - y2) * (mu2 - t2) * g2 * (1.0 - g2);
                }
                gval /= this.n;
                if (!flags[1]) {
                    continue;
                }
                if (++k > APGMaxIter) {
                    break;
                }
                gradTheta *= 2.0 / this.n;
                ThetaGrad.setEntry(0, 0, gradTheta);
            }
            ++cnt;
            fval = gval + lambda * Matlab.norm(this.W, 1);
            Printer.fprintf("Iter %d - fval: %.4f\n", cnt, fval);
            if (cnt > 1 && Math.abs(fval_pre - fval) < Matlab.eps) {
                fval_pre = fval;
            }
        } while (cnt <= maxIter);
    }
    
    @Override
    public Matrix predict(final Matrix Xt, final Matrix Tt) {
        final Matrix XtW = Xt.mtimes(this.W);
        final int n = Xt.getRowDimension();
        final Matrix PredY = new DenseMatrix(n, 1);
        for (int i = 0; i < n; ++i) {
            final double mu = XtW.getEntry(i, 0);
            final double t = Tt.getEntry(i, 0);
            final double g = 1.0 / (1.0 + Math.exp(this.rho * (t - mu)));
            PredY.setEntry(i, 0, g);
        }
        return PredY;
    }
    
    @Override
    public void loadModel(final String filePath) {
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            this.W = (Matrix)ois.readObject();
            this.rho = ois.readDouble();
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
            oos.writeObject(this.W);
            oos.writeObject(new Double(this.rho));
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
}
