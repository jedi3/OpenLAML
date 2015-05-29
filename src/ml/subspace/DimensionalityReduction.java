package ml.subspace;

import la.matrix.*;

public abstract class DimensionalityReduction
{
    protected Matrix X;
    protected Matrix R;
    protected int r;
    
    public DimensionalityReduction(final int r) {
        this.r = r;
    }
    
    public abstract void run();
    
    public void feedData(final Matrix X) {
        this.X = X;
    }
    
    public void feedData(final double[][] data) {
        this.X = new DenseMatrix(data);
    }
    
    public Matrix getDataMatrix() {
        return this.X;
    }
    
    public Matrix getReducedDataMatrix() {
        return this.R;
    }
    
    public void setReducedDimensionality(final int r) {
        this.r = r;
    }
    
    public int getReducedDimensionality() {
        return this.r;
    }
}
