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
 * @author 100024721
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
        double[] distances = Helpers.findEuclideanDistances(trainingData, instance);
        
        /* 
         * Create a list of dictionaries where each dictionary contains
         * the keys "distance" and "id".
         * The distance key stores the euclidean distance for an instance and 
         * the id key stores the hashcode for that instance object.
         */
        ArrayList<HashMap<String, Object>> table = Helpers.buildDistanceTable(trainingData, distances);

        /*
        this.sortDistanceTable(table, "distance");
        for(int i = 0; i < distances.length; i++){
            HashMap<String, Object> row = table.get(i);
            Instance inst = trainingData.get(i);
            System.out.println(row.get("distance"));
        }
        System.out.println("");
        */
        
        
        //k=5;
        /* Find the k smallest distances */
        Object[] kClosestRows = new Object[k];
        Object[] kClosestInstances = new Object[k];
        double[] classValues = new double[k];
        
        for(int i = 1; i <= k; i++){
            ArrayList<Integer> tieIndices = new ArrayList<Integer>();
            
            /* Find the positions in the table of the ith closest neighbour */
            int[] closestRowIndices = Helpers.findNthClosestNeighbour(table, i);
            
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
        //System.out.println(this.mode(this.arrayToArrayList(classValues)));
        //System.out.println(this.mode(Arrays.asList(classValues)));
        return Helpers.mode(Helpers.arrayToArrayList(classValues));
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
    
    
    
    
    
    

    
}
