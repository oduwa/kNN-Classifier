/*
 * A class containing static methods for useful functions used throughout the
 * project.
 */

package machinelearning_cw;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author 100024721
 */
public class Helpers {
    
    /**
     * Calculates the distances of the given Instance data from the
     * training data.
     * 
     * Creates an array of distances where the value at each position in the
     * array corresponds to the distance of the "instance" param from the
     * training data at that same position.
     * 
     * 
     * @param data The collection of training data instances to find distances
     *             from.
     * @param instance The data instance to find distances from
     * @return an array of distances where the value at each position in the
     * array corresponds to the distance of the "instance" param from the
     * training data at that same position.
     */
    public static double[] findEuclideanDistances(Instances data,
            Instance instance){
        /* Initialise array to hold euclidean distances */
        double[] distances = new double[data.numInstances()];
        
        /* 
         * Calculate euclidean distances between each instance of training
         * data and the instance to be classified.
         */
        for(int i = 0; i < distances.length; i++){
            Instance trainingInstance = data.get(i);
            double distance = 0;
            
            for(int j = 0; j < instance.numAttributes()-1; j++){
                double difference = trainingInstance.value(j)-instance.value(j);
                distance += Math.pow(difference, 2);
            }
            
            distances[i] = distance;
        }
        
        return distances;
    }
    
    
    /**
     * Creates table of distances of the given Instance data from the
     * training data.
     * 
     * The resulting table is in the form of an ArrayList whose elements are 
     * HashMap objects. Each item in the array list represents a row in the
     * table and each key-value pair in the HashMap represents a column.
     * 
     * The HasMap contains the keys "id" and "distance".
     * 
     * "id" stores the hash code for the training data instance and can be used
     * to refer back to the training data Instance.
     * 
     * "distance" stores the distance from the training data instance.
     * 
     * @param data The collection of training data instances.
     * @param distances An array of distances where the value at each position 
     * in the array corresponds to the distance from the training data at 
     * that position.
     * @return an ArrayList whose elements are 
     * HashMap objects. Each item in the array list represents a row in the
     * table and each key-value pair in the HashMap represents a column
     */
    public static ArrayList<HashMap<String, Object>> buildDistanceTable(
            Instances data, double[] distances){
        /* Initialise table */
        ArrayList<HashMap<String, Object>> table = 
                new ArrayList<HashMap<String, Object>>();
        
        /* Populate table */
        int i = 0;
        for(Instance eachInstance : data){
            HashMap<String, Object> row =  new HashMap<String, Object>();
            row.put("id", Integer.toHexString(eachInstance.hashCode()));
            row.put("distance", new  Double(distances[i]));
            row.put("weight", new  Double(1/(1+distances[i])));
            table.add(row);
            i++;
        }
        
        return table;
    }

    /**
     * Sorts the distance table so that the smaller data-distance pairs appear
     * above the larger ones. 
     * 
     * This in turn ranks the table, making it easier to identify the nearest
     * neighbors in terms of distance.
     * 
     * @param table A distance table in the form of an ArrayList whose 
     * elements are HashMap objects with the keys "id" and "distance".
     * @param sortParam The column of the table to sort by. Must be either 
     * "distance" or "weight"
     * 
     */
    public static void sortDistanceTable(
            ArrayList<HashMap<String, Object>> table, String sortParam){
        
        if(sortParam.equalsIgnoreCase("distance")){
            /* Sort hash map in ascending using anonymous comparator */
            Collections.sort(table, new Comparator<HashMap<String, Object>>(){

            @Override
            public int compare(HashMap<String, Object> o1,
                    HashMap<String, Object> o2) {
                if((Double) o1.get("distance") > (Double) o2.get("distance")){
                    return 1;
                }
                else if((Double)o1.get("distance") < 
                        (Double)o2.get("distance")){
                    return -1;
                }
                else{
                    return 0;
                }
            }
            
        });
        }
        else if(sortParam.equalsIgnoreCase("weight")){
            /* Sort hash map in descending order using anonymous comparator */
            Collections.sort(table, new Comparator<HashMap<String, Object>>(){

            @Override
            public int compare(HashMap<String, Object> o1, 
                    HashMap<String, Object> o2) {
                if((Double) o1.get("weight") < (Double) o2.get("weight")){
                    return 1;
                }
                else if((Double)o1.get("weight") > (Double)o2.get("weight")){
                    return -1;
                }
                else{
                    return 0;
                }
            }
            
        });
        }
        else{
            // throw exception
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
    public static int[] findNthClosestNeighbour(
            ArrayList<HashMap<String, Object>> table, int n){
        int[] result = null;
        sortDistanceTable(table, "distance");
        int count = 0;
        
        // Find first closest
        double smallestSoFar = (double) table.get(0).get("distance");
        
        /* 
         * Create 2D array where the ith element in the first array contains
         * the positions of the ith smallest values in the distance table.
         *
         * i.e. rank[0] contains an array which in turn contains the first
         * smallest value(s)
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
            if ((double) table.get(i).get("distance") == smallestSoFar) {
                rank[j][k] = i;
                k++;

            } 
            else {
                smallestSoFar = (double) table.get(i).get("distance");
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
    
    public static double distance(Instance instance1, Instance instance2){
        double distance = 0;

        for (int j = 0; j < instance1.numAttributes() - 1; j++) {
            double difference = instance1.value(j) - instance2.value(j);
            distance += Math.pow(difference, 2);
        }

        return distance;
    }
    
    /**
     * 
     * Finds the mode value in a given collection. This is the value that
     * occurs the most in the collection.
     * 
     * @param list a list of values whose mode is to be found
     * @return mode value. ie value with highest frequency.
     */
    public static double mode(ArrayList<Double> list) {
        int maxSoFar = 0;
        int maxIndex = 0;
        
        for(int i = 0; i < list.size(); i++){
            //System.out.println(list.get(i));
            Double d = list.get(i);
            int frequency = Collections.frequency(list, d);
            if(frequency > maxSoFar){
                maxSoFar = frequency;
                maxIndex = i;
            }
        }

        return list.get(maxIndex);
    }
    
    /**
     * Converts an array of doubles to an ArrayList of doubles.
     * 
     * Used in private mode method so Collections.frequency() can be used to
     * find the mode.
     * 
     * @param array the array to be converted to an ArrayList.
     * @return an ArrayList representation of the input array.
     */
    public static ArrayList<Double> arrayToArrayList(double[] array){
        ArrayList<Double> list = new ArrayList<Double>(); 
        for(int i = 0; i < array.length; i++){
            list.add(array[i]);
        }
        return list;
    }

    
    /**
     * Calculates the mean value and standard deviation for a given attribute
     * within a given dataset.
     * 
     * @param data The data whose mean and standard deviation is to be found.
     * @param attributeIndex The index of the attribute whose mean and 
     * standard deviation is to be found.
     * 
     * @return  an array of size 2 where the first index contains the mean
     * and the second index contains the standard deviation.
     */
    public static double[] meanAndStandardDeviation(Instances data,
            int attributeIndex){
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
    

    /**
     * 
     * Calculates the accuracy of a given classifier over a given dataset.
     * 
     * @param classifier The classifier to be tested.
     * @param instances The data the classifier is to be tested against.
     * 
     * @return The accuracy of the classifier.
     * @throws Exception 
     */
    public static double findClassifierAccuracy(Classifier classifier, 
            Instances instances) throws Exception{
        /* Find probablitity that predicted value is same as actual value
         * - that is, numRight/totalNum 
         */
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
    
    
    /**
     * 
     * Calculates the accuracy of a given classifier for a single instance.
     * 
     * @param classifier The classifier to be tested.
     * @param instance The data the classifier is to be tested against.
     * 
     * @return The accuracy of the classifier as either 1 or 0.
     * @throws Exception 
     */
    public static double findClassifierAccuracy(Classifier classifier,
            Instance instance) throws Exception{
        
        double result = 0;

        double prediction = classifier.classifyInstance(instance);
        double actualValue = instance.classValue();
        if (prediction == actualValue) {
            result = 1.0;
        }

        return result;
    }
}
