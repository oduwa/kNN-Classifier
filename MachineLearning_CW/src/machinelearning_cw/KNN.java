/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Odie
 */
public class KNN extends BasicKNN{

    private boolean useStandardisedAttributes = false;
    private boolean autoDetermineK = false;
    private boolean useWeightedVoting = false;
    private double mean;
    private double standardDeviation;
    private boolean isMeanAndStdDevInitialised = false;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        if(useStandardisedAttributes){
            // For each data attribute
            for(int i = 0; i < data.numAttributes() - 1; i++){
                // Calculate mean and Standard deviation
                double[] meanAndStdDev = this.meanAndStandardDeviation(data, i);
                double mean = meanAndStdDev[0];
                double stdDev = meanAndStdDev[1];
                this.mean = mean;
                this.standardDeviation = stdDev;
                isMeanAndStdDevInitialised = true;
                //System.out.println("MEAN: " + mean + "\tSTD: " + stdDev);

                // Standardise the values in all instances for given attribute
                for(Instance eachInstance : data){
                    double value = eachInstance.value(i);
                    double standardisedValue = (value - mean) / stdDev;
                    // Instead of setValue, use toDoubleArray
                    eachInstance.setValue(i, standardisedValue);
                }
            }
        }

        trainingData = new Instances(data);
        
        if(autoDetermineK){
            determineK();
        }
    }

    public void setUseStandardisedAttributes(boolean decision){
        useStandardisedAttributes = decision;
    }

    public void setAutoDetermineK(boolean autoDetermineK) {
        this.autoDetermineK = autoDetermineK;
    }

    public void setUseWeightedVoting(boolean useWeightedVoting) {
        this.useWeightedVoting = useWeightedVoting;
    }

    public void determineK() throws Exception{
        int maxK = (int) (0.4* (double)trainingData.numInstances());
        
        if(maxK > 100){
            maxK = 100;
        }
        
        int greatestAccuracyK = 1;
        double greatestAccuracy = estimateAccuracyByLOOCV(1);
        for(int k = 1; k <= maxK; k++){
            double accuracyEstimate = estimateAccuracyByLOOCV(k);
            if(accuracyEstimate > greatestAccuracy){
                greatestAccuracy = accuracyEstimate;
                greatestAccuracyK = k;
            }
        }
        
        this.k = greatestAccuracyK;
        System.out.println("DETERMINED k TO BE " + k);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if(useStandardisedAttributes){
            if(!isMeanAndStdDevInitialised){
                // throw exception
            }
            else{
                /* Standardise test instance */
                for(int i = 0; i < instance.numAttributes()-1; i++){
                    double value = instance.value(i);
                    double standardisedValue = (value - mean) / standardDeviation;
                    // TODO: Instead of setValue, use toDoubleArray
                    instance.setValue(i, standardisedValue);
                }
            }
        }
        
        if(!useWeightedVoting){
            return super.classifyInstance(instance);
        }
        else{
            /* Calculate euclidean distances */
        double[] distances = findEuclideanDistances(trainingData, instance);
        
        /* 
         * Create a list of dictionaries where each dictionary contains
         * the keys "distance", "weight" and "id".
         * The distance key stores the euclidean distance for an instance and 
         * the id key stores the hashcode for that instance object.
         */
        ArrayList<HashMap<String, Object>> table = this.buildDistanceTable(trainingData, distances);
          //k=12;  
            /* Find the k smallest distances */
        Object[] kClosestRows = new Object[k];
        Object[] kClosestInstances = new Object[k];
        double[] classValues = new double[k];

        for(int i = 1; i <= k; i++){
            ArrayList<Integer> tieIndices = new ArrayList<Integer>();
            
            /* Find the positions in the table of the ith closest neighbour */
            int[] closestRowIndices = this.findNthClosestNeighbourByWeights(table, i);

            if (closestRowIndices.length > 0) {
                /* Keep track of distance ties */
                for (int j = 0; j < closestRowIndices.length; j++) {
                    tieIndices.add(closestRowIndices[j]);
                }

                /* Break ties (by choosing winner at random) */
                Random rand = new Random();
                int matchingNeighbourPosition = tieIndices.get(rand.nextInt(tieIndices.size()));
                HashMap<String, Object> matchingRow = table.get(matchingNeighbourPosition);
                kClosestRows[i - 1] = matchingRow;
            }
        }

        /* 
         * Find the closestInstances from their rows in the table and also
         * get their class values.
         */
        for(int i = 0; i < kClosestRows.length; i++){
            /* Build up closestInstances array */
            for(int j = 0; j < trainingData.numInstances(); j++){
                Instance inst = trainingData.get(j);
                HashMap<String, Object> row = (HashMap<String, Object>)kClosestRows[i];
                if(Integer.toHexString(inst.hashCode()).equals(row.get("id"))){
                    kClosestInstances[i] = inst;
                    //System.out.println("MATCH" + inst + " " + row.get("distance"));
                }
            }
            
            /* Keep track of the class values of the closest instanes */
            Instance inst = (Instance) kClosestInstances[i];
            classValues[i] = inst.classValue();
            //System.out.println(inst + "  " +inst.classValue());
        }
        
        /* Return the most frequently occuring closest class */
        ArrayList cardsList = new ArrayList(Arrays.asList(classValues));
        return this.mode(this.arrayToArrayList(classValues));
        }
        
    }

    /*
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    */
    
    
    /************************** HELPER METHODS ***************************/
    
    private double[] meanAndStandardDeviation(Instances data, int attributeIndex){
        double sum = 0, s = 0;
        double reps = data.numInstances();
        double sumSquared = 0;
        
        for (int i = 0; i < reps; i++) {
            double value = data.get(i).value(attributeIndex);
            sum += value;
            sumSquared += (value * value);
        }
        
        double mean = sum / reps;
        double variance = sumSquared / reps - (mean * mean);
        double stdDev = Math.sqrt(variance);
        
        double[] result = {mean, stdDev};
        return result;
    }
    
    private double estimateAccuracyByThreeFoldCV(int k) throws Exception{
        ArrayList<Double> accuracies = new ArrayList<Double>();
        
        /* Partition data into s almost equal subsets. for now let s be 3 */
        int s = 3;
        int n = trainingData.size();
        ArrayList<Instances> partitions = new ArrayList<Instances>();
        for(int i = 1; i <= s; i++){
            if(i != s){
                int start = (n / s) * (i - 1);
                int stop = (n / s) * (i);
                Instances partition = new Instances(trainingData, start, n/s);
                partitions.add(partition);
            }
            else{
                int partitionedSoFar = (n/s)*(s-1);
                int remainder = n - partitionedSoFar;
                int start = (n / s) * (i - 1);
                Instances partition = new Instances(trainingData, start, remainder);
                partitions.add(partition);
            } 
        }

        /* train s classifiers and test subset i on the ith partition */
        for(int i = 0; i < s; i++){
            BasicKNN classifier = new BasicKNN();
            Instances trainingInstances = null;
            Instances testInstances = partitions.get(i);
            
            // build training data from training partitions
            for(int j = 0; j < s; j++){
                if(j != i){
                    if(trainingInstances == null){
                        trainingInstances = new Instances(partitions.get(j));
                    }
                    else{
                        trainingInstances.addAll(partitions.get(j));
                    }
                }
            }
            classifier.setK(k);
            classifier.buildClassifier(trainingInstances);
            
            /* test classifer on the one testing partition */
            double accuracy = this.findClassifierAccuracy(classifier, testInstances); 
            accuracies.add(accuracy);
        }
        
        /* find average accuracy */
        double count = accuracies.size();
        double sum = 0;
        for(Double eachAccuracy : accuracies){
            sum += eachAccuracy;
        }
        double averageAccuracy = sum/count;
        
        System.out.println("ACCURACY = " + averageAccuracy);
        return averageAccuracy;
    }
    
    private double estimateAccuracyByLOOCV(int k) throws Exception{
        ArrayList<Double> accuracies = new ArrayList<Double>();
        
        /* In a training set of n, train the model on n-1 and test on 1 */
        int n = trainingData.size();
        for(int i = 0; i < n; i++){
            Instances trainingSet = new Instances(trainingData);
            Instance testInstance = trainingSet.remove(i);
            
            BasicKNN classifier = new BasicKNN();
            classifier.setK(k);
            classifier.buildClassifier(trainingSet);
            
            /* Test classifer on test instance and measure accuracy */
            double accuracy = this.findClassifierAccuracy(classifier, testInstance); 
            accuracies.add(accuracy);
        }
        
        /* find average accuracy */
        double count = accuracies.size();
        double sum = 0;
        for(Double eachAccuracy : accuracies){
            sum += eachAccuracy;
        }
        double averageAccuracy = sum/count;
        return averageAccuracy;
    }
    
    private double findClassifierAccuracy(Classifier classifier, Instances instances) throws Exception{
        /* Find probablitity thatpredicted value is same as actual value - so numRight/totalNum */
        double numberCorrect = 0;
        double totalNumber = instances.numInstances();
        
        for(Instance instance : instances){
            double prediction = classifier.classifyInstance(instance);
            double actualValue = instance.classValue();
            if(prediction == actualValue){
                numberCorrect++;
            }  
        }
        
        return numberCorrect/totalNumber;
    }
    
    private double findClassifierAccuracy(Classifier classifier, Instance instance) throws Exception{
        
        double result = 0;

        double prediction = classifier.classifyInstance(instance);
        double actualValue = instance.classValue();
        if (prediction == actualValue) {
            result = 1.0;
        }

        return result;
    }
    
    public void testEstimateK(){
        try {
            System.out.println(estimateAccuracyByThreeFoldCV(3));
            //estimateAccuracy(3);
        } catch (Exception ex) {
            Logger.getLogger(KNN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * 
     * Find the positions in a distance table of the nth closest neighbor 
     * where n is a whole number greater than zero.
     * 
     * @param table A distance table in the form of an ArrayList whose 
     * elements are HashMap objects with the keys "id" and "distance".
     * @param n the rank of the positions to be returned. e.g. 1 for the 
     * first closest/smallest, 2 for the 2nd closest/smallest etc.
     * @return An array containing the position(s) of the nth smallest/closest
     * distances in the table. (The array has a length greater than 1 when
     * there exist ties in the distances in the table)
     */
    private int[] findNthClosestNeighbourByWeights(ArrayList<HashMap<String, Object>> table, int n){
        int[] result = null;
        sortDistanceTable(table, "weight");
        int count = 0;
        
        // Find first closest
        double largestSoFar = (double) table.get(0).get("weight");
        
        /* 
         * Create 2D array where the ith element in the first array contains
         * the positions of the ith greatest weights in the distance table.
         *
         * i.e. rank[0] contains an array which in turn contains the first
         * largest value(s)
         */
        double[][] rank = new double[table.size()][table.size()];

        /* Initialise rank matrix such that empty spaces contain -1 */
        for (int x = 0; x < rank.length; x++) {
            for (int y = 0; y < rank[x].length; y++) {
                rank[x][y] = -1;
            }
        }

        
        /* Go through the table and populate the rank matrix */
        int j = 0;
        int k = 0;
        for (int i = 0; i < table.size(); i++) {
            if ((double) table.get(i).get("weight") == largestSoFar) {
                rank[j][k] = i;
                k++;
            } 
            else {
                largestSoFar = (double) table.get(i).get("weight");
                j++;
                k = 0;
                rank[j][k] = i;
                k++;
            }
        }

        /* 
         * Calculate the size of the array to be returned based on the number 
         * of nth smallest values
         */
        for (int x = 0; x < rank[n - 1].length; x++) {
            if (rank[n - 1][x] != -1) {
                count++;
            }
        }
        
        /* initialise return variable */
        result = new int[count];
        
        /* Populate return variable accordingly */
        for (int x = 0; x < result.length; x++) {
            result[x] = (int) rank[n - 1][x];
        }

        return result;
    }
}
