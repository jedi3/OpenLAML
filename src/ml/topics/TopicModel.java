package ml.topics;

import la.matrix.*;

public abstract class TopicModel
{
    protected Matrix dataMatrix;
    protected Matrix topicMatrix;
    protected Matrix indicatorMatrix;
    public int nTopic;
    
    public TopicModel() {
        System.err.println("Number of topics undefined!");
        System.exit(1);
    }
    
    public TopicModel(final int nTopic) {
        if (nTopic < 1) {
            System.err.println("Number of topics less than one!");
            System.exit(1);
        }
        this.nTopic = nTopic;
    }
    
    public void readCorpus(final Matrix dataMatrix) {
        this.dataMatrix = dataMatrix;
    }
    
    public Matrix getTopicMatrix() {
        return this.topicMatrix;
    }
    
    public Matrix getIndicatorMatrix() {
        return this.indicatorMatrix;
    }
    
    public abstract void train();
}
