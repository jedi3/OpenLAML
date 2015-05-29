package ml.topics;

import java.io.*;
import ml.utils.*;
import java.util.*;
import java.util.regex.*;
import la.matrix.*;

public class Corpus
{
    public static int IdxStart;
    private Vector<Vector<Integer>> corpus;
    public ArrayList<TreeMap<Integer, Integer>> docTermCountArray;
    public int[][] documents;
    public int nTerm;
    public int nDoc;
    
    static {
        Corpus.IdxStart = 0;
    }
    
    public Corpus() {
        this.docTermCountArray = new ArrayList<TreeMap<Integer, Integer>>();
        this.corpus = new Vector<Vector<Integer>>();
        this.documents = null;
        this.nTerm = 0;
        this.nDoc = 0;
    }
    
    public void clearCorpus() {
        for (int i = 0; i < this.corpus.size(); ++i) {
            this.corpus.get(i).clear();
        }
        this.corpus.clear();
        this.nTerm = 0;
        this.nDoc = 0;
    }
    
    public void clearDocTermCountArray() {
        if (this.docTermCountArray.size() == 0) {
            return;
        }
        final Iterator<TreeMap<Integer, Integer>> iter = this.docTermCountArray.iterator();
        while (iter.hasNext()) {
            iter.next().clear();
        }
        this.docTermCountArray.clear();
    }
    
    public int[][] getDocuments() {
        return this.documents;
    }
    
    public void readCorpusFromLDAInputFile(final String LDAInputDataFilePath) {
        this.clearCorpus();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(LDAInputDataFilePath));
        }
        catch (FileNotFoundException e) {
            System.out.println("Cannot open file: " + LDAInputDataFilePath);
            e.printStackTrace();
        }
        String line = "";
        int termID = 0;
        int count = 0;
        int docID = 0;
        int nUniqueTerms = 0;
        final String delimiters = " :\t";
        try {
            while ((line = br.readLine()) != null) {
                ++docID;
                ++this.nDoc;
                final Vector<Integer> doc = new Vector<Integer>();
                final StringTokenizer tokenizer = new StringTokenizer(line);
                nUniqueTerms = Integer.parseInt(tokenizer.nextToken(delimiters));
                System.out.println("DocID: " + docID + ", nUniqueTerms: " + nUniqueTerms);
                while (tokenizer.hasMoreTokens()) {
                    termID = Integer.parseInt(tokenizer.nextToken(delimiters)) + (1 - Corpus.IdxStart);
                    count = Integer.parseInt(tokenizer.nextToken(delimiters));
                    for (int i = 0; i < count; ++i) {
                        doc.add(termID);
                    }
                    if (termID > this.nTerm) {
                        this.nTerm = termID;
                    }
                }
                this.corpus.add(doc);
            }
            br.close();
        }
        catch (NumberFormatException e2) {
            e2.printStackTrace();
        }
        catch (IOException e3) {
            e3.printStackTrace();
        }
        this.documents = corpus2Documents(this.corpus);
    }
    
    public void readCorpusFromDocTermCountFile(final String docTermCountFilePath) {
        this.clearDocTermCountArray();
        this.clearCorpus();
        Pattern pattern = null;
        BufferedReader br = null;
        Matcher matcher = null;
        TreeMap<Integer, Integer> docTermCountMap = null;
        Vector<Integer> doc = null;
        int docID = 0;
        int termID = 0;
        int count = 0;
        pattern = Pattern.compile("[(]([\\d]+), ([\\d]+)[)]: ([\\d]+)");
        try {
            br = new BufferedReader(new FileReader(docTermCountFilePath));
        }
        catch (FileNotFoundException e) {
            System.out.println("Cannot open file: " + docTermCountFilePath);
            e.printStackTrace();
        }
        try {
            String line;
            while ((line = br.readLine()) != null) {
                matcher = pattern.matcher(line);
                if (!matcher.find()) {
                    System.out.println("Data format for the docTermCountFile should be: (docID, termID): count");
                    System.exit(0);
                }
                docID = Integer.parseInt(matcher.group(1));
                if (docID != this.nDoc) {
                    if (this.nDoc > 0) {
                        this.docTermCountArray.add(docTermCountMap);
                        this.corpus.add(doc);
                        if (this.nTerm < docTermCountMap.lastKey()) {
                            this.nTerm = docTermCountMap.lastKey();
                        }
                        System.out.println("DocID: " + this.nDoc + ", nUniqueTerms: " + docTermCountMap.size());
                    }
                    for (int i = this.nDoc + 1; i < docID; ++i) {
                        this.docTermCountArray.add(new TreeMap<Integer, Integer>(new Utility.keyAscendComparator<Integer>()));
                        this.corpus.add(new Vector<Integer>());
                        System.out.println("DocID: " + ++this.nDoc + ", Empty");
                    }
                    docTermCountMap = new TreeMap<Integer, Integer>(new Utility.keyAscendComparator<Integer>());
                    doc = new Vector<Integer>();
                    ++this.nDoc;
                }
                termID = Integer.parseInt(matcher.group(2));
                count = Integer.parseInt(matcher.group(3));
                docTermCountMap.put(termID, count);
                for (int i = 0; i < count; ++i) {
                    doc.add(termID);
                }
            }
            if (docTermCountMap != null) {
                this.docTermCountArray.add(docTermCountMap);
                this.corpus.add(doc);
                if (this.nTerm < docTermCountMap.lastKey()) {
                    this.nTerm = docTermCountMap.lastKey();
                }
                System.out.println("DocID: " + this.nDoc + ", nUniqueTerms: " + docTermCountMap.size());
            }
            br.close();
        }
        catch (NumberFormatException e2) {
            e2.printStackTrace();
        }
        catch (IOException e3) {
            e3.printStackTrace();
        }
        this.documents = corpus2Documents(this.corpus);
    }
    
    public void readCorpusFromDocTermCountArray(final ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {
        this.clearCorpus();
        int count = 0;
        final Iterator<TreeMap<Integer, Integer>> iter = docTermCountArray.iterator();
        TreeMap<Integer, Integer> docTermCountMap = null;
        Vector<Integer> doc = null;
        while (iter.hasNext()) {
            docTermCountMap = iter.next();
            doc = new Vector<Integer>();
            ++this.nDoc;
            for (final int termID : docTermCountMap.keySet()) {
                count = docTermCountMap.get(termID);
                for (int i = 0; i < count; ++i) {
                    doc.add(termID);
                }
            }
            if (this.nTerm < docTermCountMap.lastKey()) {
                this.nTerm = docTermCountMap.lastKey();
            }
            this.corpus.add(doc);
        }
        this.documents = corpus2Documents(this.corpus);
    }
    
    public void readCorpusFromMatrix(final Matrix X) {
        this.clearCorpus();
        int count = 0;
        int termID = 0;
        Vector<Integer> doc = null;
        final int nDoc = X.getColumnDimension();
        this.nTerm = X.getRowDimension();
        if (X instanceof DenseMatrix) {
            for (int d = 0; d < nDoc; ++d) {
                doc = new Vector<Integer>();
                for (int t = 0; t < this.nTerm; ++t) {
                    count = (int)X.getEntry(t, d);
                    if (count != 0) {
                        termID = t + 1;
                        for (int i = 0; i < count; ++i) {
                            doc.add(termID);
                        }
                    }
                }
                this.corpus.add(doc);
            }
        }
        else if (X instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)X).getIr();
            jc = ((SparseMatrix)X).getJc();
            pr = ((SparseMatrix)X).getPr();
            for (int j = 0; j < nDoc; ++j) {
                doc = new Vector<Integer>();
                for (int k = jc[j]; k < jc[j + 1]; ++k) {
                    termID = ir[k] + 1;
                    count = (int)pr[k];
                    for (int l = 0; l < count; ++l) {
                        doc.add(termID);
                    }
                }
                this.corpus.add(doc);
            }
        }
        this.documents = corpus2Documents(this.corpus);
    }
    
    public static int[][] corpus2Documents(final Vector<Vector<Integer>> corpus) {
        final int[][] documents = new int[corpus.size()][];
        for (int i = 0; i < corpus.size(); ++i) {
            documents[i] = new int[corpus.get(i).size()];
            for (int w = 0; w < corpus.get(i).size(); ++w) {
                documents[i][w] = corpus.get(i).get(w) - 1;
            }
        }
        return documents;
    }
    
    public static Matrix documents2Matrix(final int[][] documents) {
        if (documents == null || documents.length == 0) {
            System.err.println("Empty documents!");
            System.exit(1);
        }
        final int N = documents.length;
        final int V = getVocabularySize(documents);
        final Matrix res = new SparseMatrix(V, N);
        int[] document = null;
        int termIdx = -1;
        for (int docIdx = 0; docIdx < documents.length; ++docIdx) {
            document = documents[docIdx];
            for (int i = 0; i < document.length; ++i) {
                termIdx = document[i];
                res.setEntry(termIdx, docIdx, res.getEntry(termIdx, docIdx) + 1.0);
            }
        }
        return res;
    }
    
    public static int getVocabularySize(final int[][] documents) {
        int maxTermIdx = 0;
        for (int i = 0; i < documents.length; ++i) {
            for (int j = 0; j < documents[i].length; ++j) {
                if (maxTermIdx < documents[i][j]) {
                    maxTermIdx = documents[i][j];
                }
            }
        }
        return maxTermIdx + 1;
    }
    
    public static void setLDATermIndexStart(final int IdxStart) {
        Corpus.IdxStart = IdxStart;
    }
}
