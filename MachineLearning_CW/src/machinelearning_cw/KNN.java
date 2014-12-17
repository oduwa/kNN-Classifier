/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author 100024721
 */
public class KNN extends BasicKNN{

    private boolean useStandardisedAttributes = true;
    private boolean autoDetermineK = false;
    private boolean useWeightedVoting = false;
    private boolean useAcceleratedNNSearch = false;
    private double[] mean;
    private double[] standardDeviation;
    private boolean isMeanAndStdDevInitialised = false;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        if(useStandardisedAttributes){
            mean = new double[data.numAttributes() - 1];
            standardDeviation = new double[data.numAttributes() - 1];
            
            // For each data attribute
            for(int i = 0; i < data.numAttributes() - 1; i++){
                // Calculate mean and Standard deviation
                double[] meanAndStdDev = Helpers.meanAndStandardDeviation(
                        data, i);
                double mean = meanAndStdDev[0];
                double stdDev = meanAndStdDev[1];
                this.mean[i] = mean;
                this.standardDeviation[i] = stdDev;
                isMeanAndStdDevInitialised = true;

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

    public void setUseAcceleratedNNSearch(boolean useAcceleratedNNSearch) {
        this.useAcceleratedNNSearch = useAcceleratedNNSearch;
    }
    
    public void determineK() throws Exception{
        ArrayList<ArrayList<HashMap<String, Object>>>distanceMatrix = null;
        
        /* Check that training data is not too large before making distance 
         * matrix as a 3000 squared arraylist of arraylists will break
         * the heap.
         *
         * If it does, use the slower approach of using a new BasicKNN instance
         * for each fold of the LOOCV.
         */
        if(trainingData.size() < 3000){
            distanceMatrix = getDistanceMatrix();
        }
        
        int maxK = (int) (0.4* (double)trainingData.numInstances());
        
        if(maxK > 100){
            maxK = 100;
        }
        
        int greatestAccuracyK = 1;
        double greatestAccuracy = 0; 
        if (distanceMatrix != null) {
            greatestAccuracy = estimateAccuracyByLOOCV(1, distanceMatrix);
        } 
        else {
            // Perform LOOCV without precomputed distance matrix
            greatestAccuracy = estimateAccuracyByLOOCV(1, trainingData);
        }
        
        for(int k = 1; k <= maxK; k++){
            double accuracyEstimate = 0;
            
            if(distanceMatrix != null){
                accuracyEstimate = estimateAccuracyByLOOCV(k, distanceMatrix);
            }
            else{
                // Perform LOOCV without precomputed distance matrix
                accuracyEstimate = estimateAccuracyByLOOCV(k, trainingData);
            }
            
            if(accuracyEstimate > greatestAccuracy){
                greatestAccuracy = accuracyEstimate;
                greatestAccuracyK = k;
            }
        }
        
        this.k = greatestAccuracyK;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // Check that classifier has been trained
        if(trainingData == null){
            throw new Exception("Classifier has not been trained."
                    + " No call to buildClassifier() was made");
        }
        
        if(useStandardisedAttributes){
            if(!isMeanAndStdDevInitialised){
                // throw exception
            }
            else{
                /* Standardise test instance */
                for(int i = 0; i < instance.numAttributes()-1; i++){
                    double value = instance.value(i);
                    double standardisedValue = 
                            (value - mean[i])/standardDeviation[i];
                    instance.setValue(i, standardisedValue);
                }
            }
        }
        
        if(!useWeightedVoting){
            return super.classifyInstance(instance);
        }
        else{
            
            if(!useAcceleratedNNSearch) {
                /* Calculate euclidean distances */
                double[] distances = Helpers.findEuclideanDistances(
                        trainingData, instance);

                /* 
                 * Create a list of dictionaries where each dictionary contains
                 * the keys "distance", "weight" and "id".
                 * The distance key stores the euclidean distance for an  
                 * instance and the id key stores the hashcode for that 
                 * instance object.
                 */
                ArrayList<HashMap<String, Object>> table = 
                        Helpers.buildDistanceTable(trainingData, distances);
  
                /* Find the k smallest distances */
                Object[] kClosestRows = new Object[k];
                Object[] kClosestInstances = new Object[k];
                double[] classValues = new double[k];

                for (int i = 1; i <= k; i++) {
                    ArrayList<Integer> tieIndices = new ArrayList<Integer>();

                   /* Find the positions in the table of the ith closest 
                    * neighbour.
                    */
                    int[] closestRowIndices = 
                            this.findNthClosestNeighbourByWeights(table, i);

                    if (closestRowIndices.length > 0) {
                        /* Keep track of distance ties */
                        for (int j = 0; j < closestRowIndices.length; j++) {
                            tieIndices.add(closestRowIndices[j]);
                        }

                        /* Break ties (by choosing winner at random) */
                        Random rand = new Random();
                        int matchingNeighbourPosition = tieIndices.get(
                                rand.nextInt(tieIndices.size()));
                        HashMap<String, Object> matchingRow = table.get(
                                matchingNeighbourPosition);
                        kClosestRows[i - 1] = matchingRow;
                    }
                }

                /* 
                 * Find the closestInstances from their rows in the table and 
                 * also get their class values.
                 */
                for (int i = 0; i < kClosestRows.length; i++) {
                    /* Build up closestInstances array */
                    for (int j = 0; j < trainingData.numInstances(); j++) {
                        Instance inst = trainingData.get(j);
                        HashMap<String, Object> row = (HashMap<String, Object>)
                                kClosestRows[i];
                        if (Integer.toHexString(
                                inst.hashCode()).equals(row.get("id"))) {
                            kClosestInstances[i] = inst;
                        }
                    }
                }

                /* Vote by weights */
                /* Get max class value */
                double[] possibleClassValues = 
                        trainingData.attributeToDoubleArray(
                                trainingData.classIndex());
                int maxClassIndex = Utils.maxIndex(possibleClassValues);
                double maxClassValue = possibleClassValues[maxClassIndex];
                ArrayList<Double> weightedVotes = new ArrayList<Double>();
                
                /* Calculate the sum of votes for each class */
                for (double i = 0; i <= maxClassValue; i++) {
                    double weightCount = 0;
                    
                    /* Calculate sum */
                    for (int j = 0; j < kClosestInstances.length; j++) {
                        Instance candidateInstance 
                                = (Instance) kClosestInstances[j];
                        if (candidateInstance.classValue() == i) {
                            // Get weight
                            HashMap<String, Object> row = 
                                (HashMap<String, Object>) kClosestRows[(int) j];
                            weightCount += (double) row.get("weight");
                        }
                    }

                    weightedVotes.add(weightCount);
                }
                
                /* Select instance with highest vote */
                Double[] votesArray = new Double[weightedVotes.size()];
                weightedVotes.toArray(votesArray);
                double greatestSoFar = votesArray[0];
                int greatestIndex = 0;
                for(int i = 0; i < votesArray.length; i++){
                    if(votesArray[i] > greatestSoFar){
                        greatestSoFar = votesArray[i];
                        greatestIndex = i;
                    }
                }

                /* 
                 * Class value will be the index because classes are indexed 
                 * from 0 upwards.
                 */
                return greatestIndex;

            }
            /* Use Orchards algorithm to accelerate NN search */
            else {
                // find k nearest neighbours
                ArrayList<Instance> nearestNeighbours = 
                        new ArrayList<Instance>();
                for (int i = 0; i < k; i++) {
                    nearestNeighbours.add(findNthClosestWithOrchards(instance, 
                            trainingData, i));
                }

                // Find their class values
                double[] classValues = new double[nearestNeighbours.size()];

                for (int i = 0; i < nearestNeighbours.size(); i++) {
                    classValues[i] = nearestNeighbours.get(i).classValue();
                }

                return Helpers.mode(Helpers.arrayToArrayList(classValues));
            }

        }
        
    }
    
    /**
     * Find the nth nearest neighbors of a data point within a data set
     * where n is a positive non zero integer.
     * 
     * @param queryPoint The data point for which the n nearest neighbor is
     * to be found.
     * @param data The data set from which the closest neighbor is to be found. 
     * @param n A positive non-zero integer representing the rank of the data
     * point to be returned.
     * @return The nth nearest neighbor of the specified query point .
     */
    public Instance findNthClosestWithOrchards(Instance queryPoint,
            Instances data, int n){
        // Pre-processing
        ArrayList<ArrayList<HashMap<String, Object>>>orchardMatrix = 
                new ArrayList<ArrayList<HashMap<String, Object>>>();
        
        // Create orchard matrix of distances
        for(Instance instance : data){
            ArrayList<HashMap<String, Object>> distancesRow = 
                    new ArrayList<HashMap<String, Object>>();
            
            // get distances for this instance
            for(Instance instance2 : data){
                double distance = Double.MAX_VALUE;
                if(!instance2.equals(instance)){
                    distance = Helpers.distance(instance, instance2);
                }
                HashMap<String, Object> entry = new HashMap<String, Object>();
                entry.put("instanceY", instance);
                entry.put("instanceX", instance2);
                entry.put("distance", (Double)distance);
                distancesRow.add(entry);
            }
            
            orchardMatrix.add(distancesRow);
            
        }

        // Sort each row
        for(ArrayList<HashMap<String, Object>> row : orchardMatrix){
            Helpers.sortDistanceTable(row, "distance");
        }
        
        HashMap<String, Object> entry = null;
        for(int i = 0; i < n; i++){
            queryPoint = (Instance) orchardsAlgorithm(orchardMatrix, queryPoint)
                    .get("instanceX");
        }
        
        return queryPoint;
    }

    /**
     * Applies Orchard's algorithm to find the nearest neighbor of a data point
     * within a data set.
     * 
     * @param orchardMatrix A matrix in the form of a 2-dimensional ArrayList
     * containing the distances from each other for a data set. Each entry in
     * the matrix contains a HashMap with the keys "instanceY", "instanceX"
     * and "distance".
     * @param queryPoint The data point for which the nearest neighbor is
     * to be found.
     * @return a HashMap entry from the provided orchardMatrix which is closest
     * to the specified queryPoint.
     */
    public HashMap<String, Object> orchardsAlgorithm(
            ArrayList<ArrayList<HashMap<String, Object>>> orchardMatrix, 
            Instance queryPoint){

        /* Let i be the index of initial guess codeword */
        int i = 0;
        int bestIndex = i;
        Instance indexedPoint = 
                (Instance) orchardMatrix.get(i).get(0).get("instanceY");
        double r = Helpers.distance(queryPoint, indexedPoint);
        int j = 0;
        Instance bestInstance = indexedPoint;
        
        double smallestDistance = Double.MAX_VALUE;
        while((double)orchardMatrix.get(i).get(j).get("distance") < 2*r){
            
            /* 
             * Check if distance between point j and query point is less than
             * r
             */
            Instance pointJ = 
                    (Instance) orchardMatrix.get(i).get(j).get("instanceX");
            double distance = Helpers.distance(queryPoint, pointJ);

            if(distance < r && distance < smallestDistance){
                bestIndex = j;
                bestInstance = pointJ;
                smallestDistance = distance;
            }
            j++;
        }

        return orchardMatrix.get(i).remove(bestIndex);
        
    }
    
    /************************** HELPER METHODS ***************************/
    
    /**
     * Computes the distances of the instances in the 
     * training data from each other.
     * 
     * @return a matrix containing the distances of the instances in the 
     * training data from each other.
     */
    private ArrayList<ArrayList<HashMap<String, Object>>> getDistanceMatrix(){
        ArrayList<ArrayList<HashMap<String, Object>>>distanceMatrix = 
                new ArrayList<ArrayList<HashMap<String, Object>>>();
        
        /* 
         * For each pattern, compute the distance between it and the entire
         * training set and storein a 2-dimensional array
         */
        for(Instance testInstance : trainingData){
            ArrayList<HashMap<String, Object>> distancesRow = 
                    new ArrayList<HashMap<String, Object>>();
            
            /* 
             * Compute distance between test pattern and rest of 
             * training data 
             */
            for(Instance trainingInstance : trainingData){
                double distance = Double.MAX_VALUE;
                if(!testInstance.equals(trainingInstance)){
                    distance = Helpers.distance(testInstance, trainingInstance);
                }
                HashMap<String, Object> entry = new HashMap<String, Object>();
                entry.put("testInstance", testInstance);
                entry.put("trainingInstance", trainingInstance);
                entry.put("distance", (Double)distance);
                distancesRow.add(entry);
            }
            
            Helpers.sortDistanceTable(distancesRow, "distance");
            distanceMatrix.add(distancesRow);
        }
        
        return distanceMatrix;
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
    private int[] findNthClosestNeighbourByWeights(
            ArrayList<HashMap<String, Object>> table, int n){
        int[] result = null;
        Helpers.sortDistanceTable(table, "weight");
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
    
    
    /**
     * 
     * Estimate the accuracy of using a value as k by applying 
     * Leave-One-Out-Cross-Validation(LOOCV).
     * 
     * @param k value of k to be tested.
     * @param distanceMatrix A matrix containing the precomputed distances of 
     * every instance of the training data from each other.
     * 
     * @return Accuracy of the calling classifier using the given value of k.
     * @throws Exception 
     */
    private double estimateAccuracyByLOOCV(int k, 
            ArrayList<ArrayList<HashMap<String, 
                    Object>>>distanceMatrix) throws Exception{
        ArrayList<Double> accuracies = new ArrayList<Double>();

        int i = 0;
        
        for(ArrayList<HashMap<String, Object>> distancesRow : distanceMatrix){
            Instance testInstance = trainingData.get(i);

            /* 
            * For each test instance, classify by choosing the 2nd to the 
            * (k+1)st in the sorted list.
            */
            Instance[] closestDistances = new Instance[k];
            double[] closestClassValues = new double[k];
            for(int j = 0; j < closestDistances.length; j++){
                closestDistances[j] = 
                        (Instance) distancesRow.get(j).get("trainingInstance");
                closestClassValues[j] = closestDistances[j].classValue();
            }
            
            /* Calculate accuracy */
            double predictedClass = 
                    Helpers.mode(Helpers.arrayToArrayList(closestClassValues));
            double actualClass = testInstance.classValue();
            double accuracy = 0;
            
            if(predictedClass == actualClass){
                accuracy = 1.0;
            }

            accuracies.add(accuracy);
            i++;
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
   
    /**
     * 
     * A slower method for estimating the accuracy of using a value as k by
     * applying Leave-One-Out-Cross-Validation(LOOCV).
     * 
     * @param k value of k to be tested.
     * @param trainingData The data against which the classifier is to be
     * tested.
     * 
     * @return Accuracy of the calling classifier using the given value of k.
     * @throws Exception 
     */
    private double estimateAccuracyByLOOCV(int k, Instances trainingData) 
            throws Exception{
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
            double accuracy = 
                    Helpers.findClassifierAccuracy(classifier, testInstance); 
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
}
