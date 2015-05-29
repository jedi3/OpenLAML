package ml.topics;

import ml.options.*;
import ml.utils.*;
import la.matrix.*;
import java.util.*;

public class LDA extends TopicModel
{
    LdaGibbsSampler gibbsSampler;
    
    public static void main(final String[] args) {
        final int[][] documents = { { 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6 }, { 2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2 }, { 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0 }, { 5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0 }, { 2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0 }, { 5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2 } };
        final LDAOptions LDAOptions = new LDAOptions();
        LDAOptions.nTopic = 2;
        LDAOptions.iterations = 5000;
        LDAOptions.burnIn = 1500;
        LDAOptions.thinInterval = 200;
        LDAOptions.sampleLag = 10;
        LDAOptions.alpha = 2.0;
        LDAOptions.beta = 0.5;
        final LDA LDA = new LDA(LDAOptions);
        LDA.readCorpus(documents);
        long start = System.currentTimeMillis();
        LDA.train();
        double elapsedTime = (System.currentTimeMillis() - start) / 1000.0;
        Printer.fprintf("Elapsed time: %.3f seconds\n\n", elapsedTime);
        Printer.fprintf("Topic--term associations: \n", new Object[0]);
        Printer.display(LDA.topicMatrix);
        Printer.fprintf("Document--topic associations: \n", new Object[0]);
        Printer.display(LDA.indicatorMatrix);
        final Matrix X = Corpus.documents2Matrix(documents);
        final TopicModel lda = new LDA(LDAOptions);
        lda.readCorpus(X);
        start = System.currentTimeMillis();
        lda.train();
        elapsedTime = (System.currentTimeMillis() - start) / 1000.0;
        Printer.fprintf("Elapsed time: %.3f seconds\n\n", elapsedTime);
        Printer.fprintf("Topic--term associations: \n", new Object[0]);
        Printer.display(lda.topicMatrix);
        Printer.fprintf("Document--topic associations: \n", new Object[0]);
        Printer.display(lda.indicatorMatrix);
    }
    
    public LDA(final LDAOptions LDAOptions) {
        super(LDAOptions.nTopic);
        this.gibbsSampler = new LdaGibbsSampler(LDAOptions);
    }
    
    public LDA() {
    }
    
    public LDA(final int nTopic) {
        super(nTopic);
    }
    
    @Override
    public void train() {
        this.gibbsSampler.run();
        this.topicMatrix = new DenseMatrix(this.gibbsSampler.getPhi()).transpose();
        this.indicatorMatrix = new DenseMatrix(this.gibbsSampler.getTheta());
    }
    
    public void readCorpus(final ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {
        this.gibbsSampler.readCorpusFromDocTermCountArray(docTermCountArray);
        this.dataMatrix = Corpus.documents2Matrix(this.gibbsSampler.documents);
    }
    
    public void readCorpus(final String LDAInputDataFilePath) {
        this.gibbsSampler.readCorpusFromLDAInputFile(LDAInputDataFilePath);
        this.dataMatrix = Corpus.documents2Matrix(this.gibbsSampler.documents);
    }
    
    public void readCorpusFromDocTermCountFile(final String docTermCountFilePath) {
        this.gibbsSampler.readCorpusFromDocTermCountFile(docTermCountFilePath);
        this.dataMatrix = Corpus.documents2Matrix(this.gibbsSampler.documents);
    }
    
    public void readCorpus(final int[][] documents) {
        if (documents == null || documents.length == 0) {
            System.err.println("Empty documents!");
            System.exit(1);
        }
        this.gibbsSampler.documents = documents;
        this.gibbsSampler.V = Corpus.getVocabularySize(documents);
        this.dataMatrix = Corpus.documents2Matrix(documents);
    }
    
    @Override
    public void readCorpus(final Matrix X) {
        this.dataMatrix = X;
        this.gibbsSampler.readCorpusFromMatrix(X);
    }
}
