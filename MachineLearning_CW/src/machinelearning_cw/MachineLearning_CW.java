/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author 100024721
 */
public class MachineLearning_CW {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        /* Initializing test datasets */
        ArrayList<Instances> trainData = new ArrayList<Instances>();
        ArrayList<Instances> testData = new ArrayList<Instances>();
        
        Instances train = WekaLoader.loadData("PitcherTrain.arff");
        Instances test = WekaLoader.loadData("PitcherTest.arff");
        trainData.add(train);
        testData.add(test);
        
        Instances bananaTrain = WekaLoader.loadData("banana-train.arff");
        Instances bananaTest = WekaLoader.loadData("banana-test.arff");
        trainData.add(bananaTrain);
        testData.add(bananaTest);
        
        Instances cloudTrain = WekaLoader.loadData("clouds-train.arff");
        Instances cloudTest = WekaLoader.loadData("clouds-test.arff");
        trainData.add(cloudTrain);
        testData.add(cloudTest);
        
        Instances concentricTrain = WekaLoader.loadData("concentric-train.arff");
        Instances concentricTest = WekaLoader.loadData("concentric-test.arff");
        trainData.add(concentricTrain);
        testData.add(concentricTest);
        
        // 3 dimensional data set
        Instances habermanTrain = WekaLoader.loadData("haberman-train.arff");
        Instances habermanTest = WekaLoader.loadData("haberman-test.arff");
        trainData.add(habermanTrain);
        testData.add(habermanTest);
        
        
        // >3 dimensional data sets
        Instances thyroidTrain = WekaLoader.loadData("thyroid-train.arff");
        Instances thyroidTest = WekaLoader.loadData("thyroid-test.arff");
        trainData.add(thyroidTrain);
        testData.add(thyroidTest);
        
        Instances heartTrain = WekaLoader.loadData("heart-train.arff");
        Instances heartTest = WekaLoader.loadData("heart-test.arff");
        trainData.add(heartTrain);
        testData.add(heartTest);
        
        Instances liverTrain = WekaLoader.loadData("liver-train.arff");
        Instances liverTest = WekaLoader.loadData("liver-test.arff");
        trainData.add(liverTrain);
        testData.add(liverTest);
        
        Instances pendigitisTrain = WekaLoader.loadData("pendigitis-train.arff");
        Instances pendigitisTest = WekaLoader.loadData("pendigitis-test.arff");
        trainData.add(pendigitisTrain);
        testData.add(pendigitisTest);
        
        Instances phonemeTrain = WekaLoader.loadData("phoneme-train.arff");
        Instances phonemeTest = WekaLoader.loadData("phoneme-test.arff");
        trainData.add(phonemeTrain);
        testData.add(phonemeTest);
        
        Instances yeastTrain = WekaLoader.loadData("yeast-train.arff");
        Instances yeastTest = WekaLoader.loadData("yeast-test.arff");
        trainData.add(yeastTrain);
        testData.add(yeastTest);
        
        
        /* Test to see that BasicKNN provides the same results obtained from
         * the hand exercise.
         */
        System.out.println("Test to see that BasicKNN provides the same"
                + " results obtained from the hand exercise:");
        System.out.println("(Ties are settled randomly)");
        BasicKNN basicKNN = new BasicKNN();
        basicKNN.buildClassifier(train);
        for(int i = 0; i < test.size(); i++){
            Instance inst = test.get(i);
            System.out.println(i+1 + ": " + basicKNN.classifyInstance(inst));
        }
        
        
        /* Initializing alternative classifiers */
        IBk wekaKNN = new IBk();
        NaiveBayes naiveBayes = new NaiveBayes();
        J48 decisionTree = new J48();
        SMO svm = new SMO();

        /* Tests for experiments 1,2 & 3 */
        KNN myKNN = new KNN(); 
        myKNN.setUseStandardisedAttributes(true);
        myKNN.setAutoDetermineK(false);
        myKNN.setUseWeightedVoting(true);
        myKNN.buildClassifier(train);
        //myKNN.setUseAcceleratedNNSearch(true);
        System.out.println("\nAccuracy Experiments:");
        MachineLearning_CW.performClassifierAccuracyTests(
                myKNN, trainData, testData, 1);
        
        /* Timing tests */
        System.out.println("\n\nTiming Experiments:");
        MachineLearning_CW.performClassifierTimingTests(
                wekaKNN, trainData, testData);
    }
    
    
    /**
     * 
     * Tests the accuracy of a classifier against a collection of datasets
     * by Resampling.
     * 
     * @param classifier The classifier to be tested
     * @param trainingDatasets A collection of Instances objects containing
     * the training data for different datasets.
     * @param testDatasets A collection of Instances objects containing
     * the test data for different datasets.
     * @param t The number of times the data should be sampled
     * @throws Exception 
     */
    public static void performClassifierAccuracyTests(Classifier classifier,
            ArrayList<Instances> trainingDatasets, ArrayList<Instances> 
                    testDatasets, int t) throws Exception{
        ArrayList<Double> accuracies = new ArrayList<Double>(); 
        Random randomGenerator = new Random();
        
        for(int i = 0; i < trainingDatasets.size(); i++){
            Instances train = trainingDatasets.get(i);
            Instances test = testDatasets.get(i);
            
            /* Test by Resampling. First, merge train and test data */
            for(int j = 0; j < t; j++){
                
                Instances mergedDataSet = mergeDataSets(train, test);
                train.clear();
                test.clear();

                /* Randomly sample n instances from the merged dataset
                 * (without replacement) to form the train set
                 */
                int n = mergedDataSet.size() / 2;
                for (int k = 0; k < n; k++) {
                    int indexToRemove = randomGenerator.nextInt(
                            mergedDataSet.size());
                    train.add(mergedDataSet.remove(indexToRemove));
                }
            
                /* Reserve remainingdata as test data */
                for(int k = 0; k < mergedDataSet.size(); k++){
                    test.add(mergedDataSet.remove(k));
                }

                /* Train classifier. Recalculates k */
                classifier.buildClassifier(train);
                
                /* Measure and record the accuracy of the classifier on
                 * the test set
                 */
                double accuracy = Helpers.findClassifierAccuracy(classifier,
                                    test);
                accuracies.add(accuracy);
            }
  
            double accuracyAverage = average(accuracies);
            System.out.println(accuracyAverage);
        }
        
    }
    
    
    /**
     * 
     * Tests the time taken for a given classifier to work against different 
     * sizes of data.
     * 
     * Prints the time taken for different values of n and the values of n
     * where n is the size of data worked on.
     * 
     * @param classifier The classifier to test
     * @param trainingDatasets A collection of all training datasets.
     * @param testDatasets A collection of all test datasets.
     * @throws Exception 
     */
    public static void performClassifierTimingTests(Classifier classifier,
            ArrayList<Instances> trainingDatasets,
            ArrayList<Instances> testDatasets) throws Exception{
        /*
         * Take the single largest data set and set up a timing experiment
         * for each classifier 
         */
        int largestIndex = 0;
        int greatestSoFar = trainingDatasets.get(0).size();
        for(int i = 0; i < trainingDatasets.size(); i++){
            if(trainingDatasets.get(i).size() > greatestSoFar){
                greatestSoFar = trainingDatasets.get(i).size();
                largestIndex = i;
            }
        }

        /*
         * Time classifier for different train set sizes with fixed 
         * test set size
         */
        int cap = 0;
        for(int n = 100; n < trainingDatasets.get(largestIndex).size(); n+=300){ 
            Instances timeTrain = new Instances(
                    trainingDatasets.get(largestIndex), 0, n);
            Instances timeTest = testDatasets.get(largestIndex);
            MachineLearning_CW.timeClassifier(
                    classifier, timeTrain, timeTest, 100);
            cap = n;
            
            /* Run the experiment for n = 100 twice, to offset the effects of
             * the Java Garbage Collector and caching.
             */
            if(n == 100){
                MachineLearning_CW.timeClassifier(
                        classifier, timeTrain, timeTest, 100);
            }
        }
        
        System.out.println("FOR n = 100 : INCREMENT BY 300 : CAP AT " + cap);
    }
    
    /**
     * 
     * Tests the speed of a classifier in milliseconds and prints it. 
     * This is achieved by checking the time taken for the given classifier to
     * classify some given data.
     * 
     * 
     * @param classifier The classifier to be tested.
     * @param train The data with which to train the classifier.
     * @param test The data the classifier is to be tested against.
     * @param t The number of times the test should be carried out and averaged.
     * @throws Exception 
     */
    public static void timeClassifier(Classifier classifier, Instances train,
            Instances test, int t) throws Exception{
        
        ArrayList<Double> times = new ArrayList<Double>(); 
        
        /* Carry out test t+1 times and average.
         * The first run is ignored to offset the effects of
         * the Java Garbage Collector and caching.
         */
        for (int i = 0; i < t+1; i++) {
            // Time the build and classifyInstance methods
            double t1 = System.nanoTime();

            classifier.buildClassifier(train);
            for (Instance eachInstance : test) {
                classifier.classifyInstance(eachInstance);
            }

            double t2 = System.nanoTime() - t1;

            // Convert to ms
            double timeTaken = t2 / 1000000.0;
            
            if(i != 0){
                times.add(timeTaken);
            }  
        }

        double averageTime = average(times);
        System.out.println(averageTime);   
    }
    
    public static Instances mergeDataSets(Instances datasetA, 
            Instances datasetB){
        Instances mergedDataSet = new Instances(datasetA);
        
        for(Instance inst : datasetB){
            mergedDataSet.add(inst);
        }
        
        return mergedDataSet;
    }
    
    public static double average(ArrayList<Double> numbers){
        
        double sum = 0;
        int count = 0;
        
        for(int i = 0; i < numbers.size(); i++){
            sum += numbers.get(i).doubleValue();
            count++;
        }
        
        return sum/count;
        
    }
    
}
