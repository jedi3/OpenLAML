package ml.classification;

import java.io.*;
import la.matrix.*;

class MaxEntModel implements Serializable
{
    private static final long serialVersionUID = 8767272469004168519L;
    public int nClass;
    public int nFeature;
    Matrix W;
    int[] IDLabelMap;
    
    public MaxEntModel(final int nClass, final Matrix W, final int[] IDLabelMap) {
        this.nClass = nClass;
        this.W = W;
        this.IDLabelMap = IDLabelMap;
        this.nFeature = W.getRowDimension();
    }
}
