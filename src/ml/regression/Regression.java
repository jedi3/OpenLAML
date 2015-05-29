package ml.regression;

import ml.options.*;
import la.matrix.*;
import ml.utils.*;

public abstract class Regression
{
    public int ny;
    public int p;
    public int n;
    public Matrix X;
    public Matrix Y;
    public Matrix W;
    public double epsilon;
    public int maxIter;
    
    public Regression() {
        this.ny = 0;
        this.p = 0;
        this.n = 0;
        this.X = null;
        this.Y = null;
        this.W = null;
        this.epsilon = 1.0E-6;
        this.maxIter = 600;
    }
    
    public Regression(final double epsilon) {
        this.ny = 0;
        this.p = 0;
        this.n = 0;
        this.X = null;
        this.Y = null;
        this.W = null;
        this.epsilon = epsilon;
        this.maxIter = 600;
    }
    
    public Regression(final int maxIter, final double epsilon) {
        this.ny = 0;
        this.p = 0;
        this.n = 0;
        this.X = null;
        this.Y = null;
        this.W = null;
        this.epsilon = epsilon;
        this.maxIter = maxIter;
    }
    
    public Regression(final Options options) {
        this.ny = 0;
        this.p = 0;
        this.n = 0;
        this.X = null;
        this.Y = null;
        this.W = null;
        this.epsilon = options.epsilon;
        this.maxIter = options.maxIter;
    }
    
    public void feedData(final Matrix X) {
        this.X = X;
        this.p = X.getColumnDimension();
        this.n = X.getRowDimension();
        if (this.Y != null && X.getRowDimension() != this.Y.getRowDimension()) {
            System.err.println("The number of dependent variable vectors and the number of data samples do not match!");
            System.exit(1);
        }
    }
    
    public void feedData(final double[][] data) {
        this.feedData(new DenseMatrix(data));
    }
    
    public void feedDependentVariables(final Matrix Y) {
        this.Y = Y;
        this.ny = Y.getColumnDimension();
        if (this.X != null && Y.getRowDimension() != this.n) {
            System.err.println("The number of dependent variable vectors and the number of data samples do not match!");
            System.exit(1);
        }
    }
    
    public void feedDependentVariables(final double[][] depVars) {
        this.feedDependentVariables(new DenseMatrix(depVars));
    }
    
    public abstract void train();
    
    public abstract void train(final Matrix p0);
    
    public abstract Matrix train(final Matrix p0, final Matrix p1);
    
    public abstract Matrix train(final Matrix p0, final Matrix p1, final Matrix p2);
    
    public Matrix predict(final Matrix Xt) {
        if (Xt.getColumnDimension() != this.p) {
            System.err.println("Dimensionality of the test data doesn't match with the training data!");
            System.exit(1);
        }
        if (this instanceof LinearRegression) {
            return Xt.mtimes(this.W).plus(Matlab.repmat(new DenseMatrix(((LinearRegression)this).B, 2), 3, 1));
        }
        return Xt.mtimes(this.W);
    }
    
    public Matrix predict(final double[][] Xt) {
        return this.predict(new DenseMatrix(Xt));
    }
    
    public abstract void loadModel(final String p0);
    
    public abstract void saveModel(final String p0);
}
