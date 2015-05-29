package ml.utils;

public class FindResult
{
    public int[] rows;
    public int[] cols;
    public double[] vals;
    
    public FindResult(final int[] rows, final int[] cols, final double[] vals) {
        this.rows = rows;
        this.cols = cols;
        this.vals = vals;
    }
}
