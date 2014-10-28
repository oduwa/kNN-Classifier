/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Odie
 */
public class BasicKNN implements Classifier{

    protected int k = 1;
    protected Instances trainingData;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainingData = new Instances(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        /* Calculate euclidean distances */
        double[] distances = findEuclideanDistances(trainingData, instance);
        
        /* 
         * Create a list of dictionaries where each dictionary contains
         * the keys "distance" and "id".
         * The distance key stores the euclidean distance for an instance and 
         * the id key stores the hashcode for that instance object.
         */
        ArrayList<HashMap<String, Object>> table = this.buildDistanceTable(trainingData, distances);

        
        for(int i = 0; i < distances.length; i++){
            HashMap<String, Object> row = table.get(i);
            Instance inst = trainingData.get(i);
            System.out.println(row.get("distance"));
        }
        System.out.println("");
        //k=5;
        
        
        //k=5;
        /* Find the k smallest distances */
        Object[] kClosestRows = new Object[k];
        Object[] kClosestInstances = new Object[k];
        double[] classValues = new double[k];
        
        for(int i = 1; i <= k; i++){
            ArrayList<Integer> tieIndices = new ArrayList<Integer>();
            
            /* Find the positions in the table of the ith closest neighbour */
            int[] closestRowIndices = this.findNthClosestNeighbour(table, i);
            
            /* Keep track of distance ties */
            for(int j = 0; j < closestRowIndices.length; j++) {
                tieIndices.add(closestRowIndices[j]);
            }
            
            /* Break ties (by choosing winner at random) */
            Random rand = new Random();
            int matchingNeighbourPosition = tieIndices.get(rand.nextInt(tieIndices.size()));
            HashMap<String, Object> matchingRow = table.get(matchingNeighbourPosition);
            kClosestRows[i-1] = matchingRow;
            //System.out.println("POS: " + matchingNeighbourPosition + " DIST: " + matchingRow);
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
        //System.out.println(this.mode(this.arrayToArrayList(classValues)));
        //System.out.println(this.mode(Arrays.asList(classValues)));
        return this.mode(this.arrayToArrayList(classValues));
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public void setK(int k){
        if(k <= 0 && k >= trainingData.numInstances()){
            throw new IllegalArgumentException("k Must be an integer value "
                    + "greater than 0");
        }
        else{
            this.k = k;
        }
    }
    
    
    /************************ HELPER METHODS **************************/
    
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
    private double[] findEuclideanDistances(Instances data, Instance instance){
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
     * @param distances An array of distances where the value at each position in the
     * array corresponds to the distance from the training data at that position
     * @return an ArrayList whose elements are 
     * HashMap objects. Each item in the array list represents a row in the
     * table and each key-value pair in the HashMap represents a column
     */
    private ArrayList<HashMap<String, Object>> buildDistanceTable(Instances data, double[] distances){
        /* Initialise table */
        ArrayList<HashMap<String, Object>> table = new ArrayList<HashMap<String, Object>>();
        
        /* Populate table */
        int i = 0;
        for(Instance eachInstance : data){
            HashMap<String, Object> row =  new HashMap<String, Object>();
            row.put("id", Integer.toHexString(eachInstance.hashCode()));
            row.put("distance", new  Double(distances[i]));
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
     * neighbours in terms of distance.
     * 
     * @param table A distance table in the form of an ArrayList whose 
     * elements are HashMap objects with the keys "id" and "distance".
     * 
     */
    private void sortDistanceTable(ArrayList<HashMap<String, Object>> table){
        /* Sort hash map using anonymous comparator */
        Collections.sort(table, new Comparator<HashMap<String, Object>>(){

            @Override
            public int compare(HashMap<String, Object> o1, HashMap<String, Object> o2) {
                if((Double) o1.get("distance") > (Double) o2.get("distance")){
                    return 1;
                }
                else if((Double)o1.get("distance") < (Double)o2.get("distance")){
                    return -1;
                }
                else{
                    return 0;
                }
            }
            
        });
    }
    
    
    /**
     * 
     * Find the positions in a distance table of the nth closest neighbour 
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
    private int[] findNthClosestNeighbour(ArrayList<HashMap<String, Object>> table, int n){
        int[] result = null;
        sortDistanceTable(table);
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
    
    
    /**
     * 
     * Finds the mode value in a given collection. This is the value that
     * occurs the most in the collection.
     * 
     * @param list a list of values whose mode is to be found
     * @return mode value. ie value with highest frequency.
     */
    private double mode(ArrayList<Double> list) {
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
    private ArrayList<Double> arrayToArrayList(double[] array){
        ArrayList<Double> list = new ArrayList<Double>(); 
        for(int i = 0; i < array.length; i++){
            list.add(array[i]);
        }
        return list;
    }

    
}
