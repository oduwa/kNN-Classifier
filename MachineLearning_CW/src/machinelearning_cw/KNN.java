/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import java.util.ArrayList;
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

    private boolean useStandardisedAttributes = true;
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
        
        return super.classifyInstance(instance);
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
}
