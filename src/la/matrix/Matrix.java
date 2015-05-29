package la.matrix;

import la.vector.*;

public interface Matrix
{
    int getRowDimension();
    
    int getColumnDimension();
    
    double getEntry(int p0, int p1);
    
    void setEntry(int p0, int p1, double p2);
    
    Matrix getSubMatrix(int p0, int p1, int p2, int p3);
    
    Matrix getSubMatrix(int[] p0, int[] p1);
    
    Matrix getRows(int p0, int p1);
    
    Matrix getRows(int... p0);
    
    Vector[] getRowVectors(int p0, int p1);
    
    Vector[] getRowVectors(int... p0);
    
    Matrix getRowMatrix(int p0);
    
    void setRowMatrix(int p0, Matrix p1);
    
    Vector getRowVector(int p0);
    
    void setRowVector(int p0, Vector p1);
    
    Matrix getColumns(int p0, int p1);
    
    Matrix getColumns(int... p0);
    
    Vector[] getColumnVectors(int p0, int p1);
    
    Vector[] getColumnVectors(int... p0);
    
    Matrix getColumnMatrix(int p0);
    
    void setColumnMatrix(int p0, Matrix p1);
    
    Vector getColumnVector(int p0);
    
    void setColumnVector(int p0, Vector p1);
    
    Matrix mtimes(Matrix p0);
    
    Matrix times(Matrix p0);
    
    Matrix times(double p0);
    
    Matrix plus(Matrix p0);
    
    Matrix plus(double p0);
    
    Matrix minus(Matrix p0);
    
    Matrix minus(double p0);
    
    Matrix transpose();
    
    Matrix copy();
    
    Vector operate(Vector p0);
    
    void clear();
    
    double[][] getData();
}
