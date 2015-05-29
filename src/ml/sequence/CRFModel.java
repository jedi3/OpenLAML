package ml.sequence;

import java.io.*;
import la.vector.*;

class CRFModel implements Serializable
{
    private static final long serialVersionUID = -2734854735411482584L;
    int numStates;
    int startIdx;
    int d;
    Vector W;
    
    public CRFModel(final int numStates, final int startIdx, final Vector W) {
        this.numStates = numStates;
        this.W = W;
        this.startIdx = startIdx;
        this.d = W.getDim();
    }
}
