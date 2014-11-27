/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

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
        
        ArrayList<Instances> trainData = new ArrayList<Instances>();
        ArrayList<Instances> testData = new ArrayList<Instances>();
        
        Instances train = WekaLoader.loadData("PitcherTrain.arff");
        Instances test = WekaLoader.loadData("PitcherTest.arff");
        //trainData.add(train);
        //testData.add(test);
        
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
        
        
        // >2 dimensional data sets
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
        
        
        /*
        for(Instance inst : cloudTrain){
            if(inst.classValue() == 1){
                System.out.println(inst.value(1));
            }  
        }
        */
        
        /*
        BasicKNN basicKNN = new BasicKNN();
        basicKNN.buildClassifier(train);
        basicKNN.classifyInstance(test.firstInstance());
         */
        
        
        //System.out.println(train + "\n\n\n");
        //KNN knn = new KNN();
        //knn.setUseStandardisedAttributes(false);
        //knn.setAutoDetermineK(true);
        //knn.setUseWeightedVoting(true);
        //knn.setUseAcceleratedNNSearch(true);
        //knn.buildClassifier(train);
        //knn.testEstimateK();
        //System.out.println("DECISION: " + knn.classifyInstance(test.get(0)));
        //*/
        //System.out.println(knn.findNClosestNeighbourWithOrchards(test.get(0), train, 3));
        
        //System.out.println(knn.orchardsAlgorithm(test.get(1), train));;
        
        
        //IBk wekaKNN = new IBk(); wekaKNN.buildClassifier(bananaTrain);
        KNN myKNN = new KNN(); 
        //myKNN.setUseStandardisedAttributes(true);
        //myKNN.setAutoDetermineK(false);
        myKNN.setAutoDetermineK(true);
        //myKNN.setUseWeightedVoting(true);
        //MachineLearning_CW.performClassifierAccuracyTests(myKNN, trainData, testData, 1);
        System.out.println(trainData.get(4).size());
        myKNN.buildClassifier(trainData.get(4));
        System.out.println(Helpers.findClassifierAccuracy(myKNN, testData.get(4)));
        
        
        
        //0.8687040181097906, 0.8670062252405206
        //myKNN.setUseStandardisedAttributes(true);//myKNN.setUseWeightedVoting(true);//myKNN.setAutoDetermineK(true);myKNN.setUseAcceleratedNNSearch(true);
//        myKNN.buildClassifier(train);knn.buildClassifier(train);
//        int i = 0;
//        for(Instance inst : test){
//            //System.out.println("WEKA: " + wekaKNN.classifyInstance(inst) + " ME: " + myKNN.classifyInstance(inst));
//            double wekaAnswer = wekaKNN.classifyInstance(inst);
//            double myAnswer = myKNN.classifyInstance(inst);
//            if(wekaAnswer != myAnswer){
//                System.out.println("THERES A MISMATCH AT INDEX " + i);
//                System.out.println("WEKA: " + wekaAnswer + " ME: " + myAnswer);
//                System.out.println("VERDICT: " + inst.classValue());
//            }
//            i++;
//        }
        //1070
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
    public static void performClassifierAccuracyTests(Classifier classifier, ArrayList<Instances> trainingDatasets, ArrayList<Instances> testDatasets, int t) throws Exception{
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
                    int indexToRemove = randomGenerator.nextInt(mergedDataSet.size());
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
                double accuracy = Helpers.findClassifierAccuracy(classifier, test);
                accuracies.add(accuracy);
            }
  
            double accuracyAverage = average(accuracies);
            System.out.println(accuracyAverage);
            
        }
        
    }
    
    /**
     * 
     * Tests the speed of a classifier. This is achieved by checking the time
     * taken for the given classifier to classify some given data.
     * 
     * @param classifier The classifier to be tested.
     * @param train The data with which to train the classifier.
     * @param test The data the classifier is to be tested against.
     * @throws Exception 
     */
    public static void performClassifierTimingTests(Classifier classifier, Instances train, Instances test) throws Exception{
        // Time the build and classifyInstance methods
        long t1 = System.nanoTime();
        
        classifier.buildClassifier(train);
        for(Instance eachInstance : test){
            classifier.classifyInstance(eachInstance);
        }
        
        long t2 = System.nanoTime() - t1;
        
        System.out.println(t2);   
    }
    
    public static Instances mergeDataSets(Instances datasetA, Instances datasetB){
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
