package ml.temporal;

import ml.options.*;
import ml.optimization.*;
import ml.utils.*;
import ml.regression.*;
import la.matrix.*;
import java.io.*;

public class TemporalLASSO extends SurvivalScoreModel
{
    private int gType;
    private double theta;
    private double rho;
    private double mu;
    private Options options;
    
    public static void main(final String[] args) {
    }
    
    public TemporalLASSO() {
    }
    
    public TemporalLASSO(final Options options, final int gType) {
        this.options = options;
        this.gType = gType;
    }
    
    public void initialize(final double theta0, final double rho0, final double mu0) {
        this.theta = theta0;
        this.rho = rho0;
        this.mu = mu0;
    }
    
    @Override
    public void initialize(final double... params) {
        if (params.length == 3) {
            this.theta = params[0];
            this.rho = params[1];
            this.mu = params[2];
        }
        else if (params.length == 2) {
            this.rho = params[0];
            this.mu = params[1];
        }
        else if (params.length == 1) {
            this.theta = params[0];
        }
    }
    
    public void setParams(final double... params) {
        if (params.length == 3) {
            this.theta = params[0];
            this.rho = params[1];
            this.mu = params[2];
        }
        else if (params.length == 2) {
            this.rho = params[0];
            this.mu = params[1];
        }
        else if (params.length == 1) {
            this.theta = params[0];
        }
    }
    
    @Override
    public void train() {
        final int ny = this.Y.getColumnDimension();
        final int maxIter = this.options.maxIter;
        if (Double.isInfinite(this.theta)) {
            this.theta = Matlab.sumAll(this.Y) / this.n;
        }
        if (Double.isInfinite(this.rho)) {
            this.rho = 1.0;
        }
        if (Double.isInfinite(this.mu)) {
            this.mu = 2.5;
        }
        double[] params = null;
        final Matrix Xg = this.X.copy();
        final double lambda = this.options.lambda;
        final Options options = new Options();
        options.maxIter = 50;
        options.lambda = this.n * lambda;
        options.verbose = this.options.verbose;
        options.calc_OV = this.options.calc_OV;
        options.epsilon = this.options.epsilon;
        final Regression lasso = new LASSO(options);
        this.W = Matlab.zeros(this.p, ny);
        Matrix Theta = null;
        final Matrix XW = Matlab.zeros(Matlab.size(this.Y));
        AcceleratedProximalGradient.prox = new ProxPlus();
        AcceleratedProximalGradient.type = 0;
        Matrix ThetaGrad = null;
        switch (this.gType) {
            case 1: {
                params = new double[] { this.theta };
                Theta = Matlab.ones(1, 1);
                ThetaGrad = Matlab.ones(1, 1);
                break;
            }
            case 2: {
                params = new double[] { this.rho, this.mu };
                Theta = Matlab.ones(2, 1);
                ThetaGrad = Matlab.ones(2, 1);
                break;
            }
            case 3: {
                params = new double[] { this.theta };
                Theta = Matlab.ones(1, 1);
                ThetaGrad = Matlab.ones(1, 1);
                break;
            }
            case 5: {
                params = new double[] { this.rho, this.mu };
                Theta = Matlab.ones(2, 1);
                ThetaGrad = Matlab.ones(2, 1);
                break;
            }
            case 6: {
                params = new double[] { this.theta };
                Theta = Matlab.ones(1, 1);
                ThetaGrad = Matlab.ones(1, 1);
                break;
            }
            case 7: {
                params = new double[] { this.rho };
                Theta = Matlab.ones(1, 1);
                ThetaGrad = Matlab.ones(1, 1);
                break;
            }
        }
        double fval = 0.0;
        double fval_pre = 0.0;
        int cnt = 0;
        do {
            Matrix gVal = g(this.gType, this.T, params);
            InPlaceOperator.mtimes(Xg, Matlab.diag(gVal), this.X);
            this.W = lasso.train(Xg, this.Y, this.W);
            boolean[] flags = null;
            final double epsilon = 0.001;
            int k = 0;
            double gval = 0.0;
            final int APGMaxIter = 1000;
            final double hval = 0.0;
            InPlaceOperator.mtimes(XW, this.X, this.W);
            InPlaceOperator.mtimes(ThetaGrad, dg(this.gType, this.T, params), XW.times(g(this.gType, this.T, params)).minus(this.Y).times(XW));
            InPlaceOperator.timesAssign(ThetaGrad, 2.0 / this.n);
            gval = Matlab.norm(this.Y.minus(Xg.mtimes(this.W)), "fro");
            gval = gval * gval / this.n;
            Theta.setEntry(0, 0, params[0]);
            if (this.gType == 2 || this.gType == 5) {
                Theta.setEntry(1, 0, params[1]);
            }
            while (true) {
                flags = AcceleratedProximalGradient.run(ThetaGrad, gval, hval, epsilon, Theta);
                if (flags[0]) {
                    break;
                }
                params[0] = Theta.getEntry(0, 0);
                if (this.gType == 2 || this.gType == 5) {
                    params[1] = Theta.getEntry(1, 0);
                }
                gVal = g(this.gType, this.T, params);
                InPlaceOperator.mtimes(Xg, Matlab.diag(gVal), this.X);
                gval = Matlab.norm(this.Y.minus(Xg.mtimes(this.W)), "fro");
                gval = gval * gval / this.n;
                if (!flags[1]) {
                    continue;
                }
                if (++k > APGMaxIter) {
                    break;
                }
                InPlaceOperator.mtimes(ThetaGrad, dg(this.gType, this.T, params), XW.times(g(this.gType, this.T, params)).minus(this.Y).times(XW));
                InPlaceOperator.timesAssign(ThetaGrad, 2.0 / this.n);
            }
            params[0] = Theta.getEntry(0, 0);
            if (this.gType == 2 || this.gType == 5) {
                params[1] = Theta.getEntry(1, 0);
            }
            ++cnt;
            fval = gval + lambda * Matlab.norm(this.W, 1);
            Printer.fprintf("Iter %d - fval: %.4f\n", cnt, fval);
            if (cnt > 1 && Math.abs(fval_pre - fval) < Matlab.eps) {
                break;
            }
            fval_pre = fval;
        } while (cnt <= maxIter);
        switch (this.gType) {
            case 1: {
                this.theta = params[0];
                break;
            }
            case 2: {
                this.rho = params[0];
                this.mu = params[1];
                break;
            }
            case 3: {
                this.theta = params[0];
                break;
            }
            case 5: {
                this.rho = params[0];
                this.mu = params[1];
                break;
            }
            case 6: {
                this.theta = params[0];
                break;
            }
            case 7: {
                this.rho = params[0];
                break;
            }
        }
    }
    
    static Matrix dg(final int gType, final Matrix T, final double[] params) {
        final int n = T.getRowDimension();
        double[][] resData = null;
        switch (gType) {
            case 1: {
                resData = new double[1][n];
                resData[0] = new double[n];
                final double theta = params[0];
                final Matrix TSq = T.times(T);
                final Matrix TCu = Matlab.pow(T, 3.0);
                final double thetaCu = Math.pow(theta, 3.0);
                for (int i = 0; i < T.getRowDimension(); ++i) {
                    final double tSq = TSq.getEntry(i, 0);
                    final double tCu = TCu.getEntry(i, 0);
                    resData[0][i] = 6.0 * tSq * (tSq - thetaCu) / Math.pow(thetaCu + 2.0 * tCu, 2.0);
                }
                break;
            }
            case 2: {
                resData = new double[2][n];
                resData[0] = new double[n];
                resData[1] = new double[n];
                final double rho = params[0];
                final double mu = params[1];
                for (int j = 0; j < T.getRowDimension(); ++j) {
                    final double t = T.getEntry(j, 0);
                    final double emrhot = Math.exp(-rho * t);
                    resData[0][j] = t * emrhot * Math.cos(mu * t);
                    resData[1][j] = t * emrhot * Math.sin(mu * t);
                }
                break;
            }
            case 3: {
                resData = new double[1][n];
                resData[0] = new double[n];
                final double theta = params[0];
                final Matrix TSq = T.times(T);
                for (int k = 0; k < T.getRowDimension(); ++k) {
                    final double tsq = TSq.getEntry(k, 0);
                    resData[0][k] = -tsq / Math.pow(theta + tsq, 2.0);
                }
                break;
            }
            case 5: {
                resData = new double[2][n];
                resData[0] = new double[n];
                resData[1] = new double[n];
                final double rho = params[0];
                final double mu = params[1];
                for (int j = 0; j < T.getRowDimension(); ++j) {
                    final double t = T.getEntry(j, 0);
                    final double tmmu = t - mu;
                    final double erhotmmu = Math.exp(rho * tmmu);
                    final double denominator = Math.pow(1.0 + Math.exp(rho * (t - mu)), 2.0);
                    resData[0][j] = -tmmu * erhotmmu / denominator;
                    resData[1][j] = rho * erhotmmu / denominator;
                }
                break;
            }
            case 6: {
                resData = new double[1][n];
                resData[0] = new double[n];
                final double theta = params[0];
                final Matrix TSq = T.times(T);
                for (int k = 0; k < T.getRowDimension(); ++k) {
                    final double tsq = TSq.getEntry(k, 0);
                    resData[0][k] = tsq / Math.pow(theta + tsq, 2.0);
                }
                break;
            }
            case 7: {
                resData = new double[1][n];
                resData[0] = new double[n];
                final double rho = params[0];
                for (int l = 0; l < T.getRowDimension(); ++l) {
                    final double t2 = T.getEntry(l, 0);
                    resData[0][l] = -t2 * Math.exp(-rho * t2);
                }
                break;
            }
        }
        return new DenseMatrix(resData);
    }
    
    static Matrix g(final int gType, final Matrix T, final double[] params) {
        final Matrix res = T.copy();
        switch (gType) {
            case 1: {
                final double theta = params[0];
                final Matrix TSq = T.times(T);
                final Matrix TCu = Matlab.pow(T, 3.0);
                final double thetaCu = Math.pow(theta, 3.0);
                for (int i = 0; i < T.getRowDimension(); ++i) {
                    final double tSq = TSq.getEntry(i, 0);
                    final double tCu = TCu.getEntry(i, 0);
                    res.setEntry(i, 0, 3.0 * theta * tSq / (thetaCu + 2.0 * tCu));
                }
                break;
            }
            case 2: {
                final double rho = params[0];
                final double mu = params[1];
                for (int j = 0; j < T.getRowDimension(); ++j) {
                    final double t = T.getEntry(j, 0);
                    res.setEntry(j, 0, 1.0 - Math.exp(-rho * t) * Math.cos(mu * t));
                }
                break;
            }
            case 3: {
                final double theta = params[0];
                final Matrix TSq = T.times(T);
                for (int k = 0; k < T.getRowDimension(); ++k) {
                    final double tsq = TSq.getEntry(k, 0);
                    res.setEntry(k, 0, tsq / (theta + tsq));
                }
                break;
            }
            case 5: {
                final double rho = params[0];
                final double mu = params[1];
                for (int j = 0; j < T.getRowDimension(); ++j) {
                    final double t = T.getEntry(j, 0);
                    res.setEntry(j, 0, 1.0 / (1.0 + Math.exp(rho * (t - mu))));
                }
                break;
            }
            case 6: {
                final double theta = params[0];
                final Matrix TSq = T.times(T);
                for (int k = 0; k < T.getRowDimension(); ++k) {
                    final double tsq = TSq.getEntry(k, 0);
                    res.setEntry(k, 0, theta / (theta + tsq));
                }
                break;
            }
            case 7: {
                final double rho = params[0];
                for (int l = 0; l < T.getRowDimension(); ++l) {
                    final double t2 = T.getEntry(l, 0);
                    res.setEntry(l, 0, Math.exp(-rho * t2));
                }
                break;
            }
        }
        return res;
    }
    
    @Override
    public Matrix predict(final Matrix Xt, final Matrix Tt) {
        double[] params = new double[2];
        switch (this.gType) {
            case 1: {
                params = new double[] { this.theta };
                break;
            }
            case 2: {
                params = new double[] { this.rho, this.mu };
                break;
            }
            case 3: {
                params = new double[] { this.theta };
                break;
            }
            case 5: {
                params = new double[] { this.rho, this.mu };
                break;
            }
            case 6: {
                params = new double[] { this.theta };
                break;
            }
            case 7: {
                params = new double[] { this.rho };
                break;
            }
        }
        final Matrix gVal = g(this.gType, Tt, params);
        final Matrix PredY = Xt.mtimes(this.W).times(gVal);
        return PredY;
    }
    
    @Override
    public void loadModel(final String filePath) {
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            this.W = (Matrix)ois.readObject();
            final double[] params = (double[])ois.readObject();
            this.setParams(params);
            this.gType = (int)ois.readObject();
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
            oos.writeObject(new double[] { this.theta, this.rho, this.mu });
            oos.writeObject(new Integer(this.gType));
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
